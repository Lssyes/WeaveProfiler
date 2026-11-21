#!/usr/bin/env python3
"""把 detector / validator 的判定结果渲染成修补 prompt。

第 1 步缺件与第 3 步不达标共用同一份骨架（4_validate/repair/repair_prompt.md）与
同一张路由表（knowledge/templates/repair_routes.json），只是入参不同：
  --detect   detect_instrumentation.py 的 verdict JSON  -> 存在性缺口
  --validate validate_capability.py --json 的输出        -> 有效性缺口

事实段一律从 detector 的 transport / clock / signals 现取，探不出的显式写
「未探明」而不是留空 —— 假装知道比承认不知道危险。
"""
import argparse
import json
import os
import sys

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SKELETON = os.path.join(SCRIPT_DIR, "repair_prompt.md")
DEFAULT_ROUTES = os.path.join(SCRIPT_DIR, "repair_routes.json")

LANG_BY_EXT = {
    ".cu": "cuda_cpp",
    ".cuh": "cuda_cpp",
    ".cc": "cuda_cpp",
    ".cpp": "cuda_cpp",
    ".h": "cuda_cpp",
    ".hpp": "cuda_cpp",
}

UNKNOWN = "未探明 —— 你自行确定后必须回填 <项目>/profile/abi/*.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def guess_lang(detect, override):
    if override:
        return override
    files = list(detect.get("signals", {}).get("fixed_slot_files") or [])
    for entry in detect.get("signals", {}).get("coupled_decoder_candidates") or []:
        files.append(entry.get("file", ""))
    for name in files:
        ext = os.path.splitext(name)[1].lower()
        if ext in LANG_BY_EXT:
            return LANG_BY_EXT[ext]
    # CuTe DSL 与 Triton 都是 .py，靠 kernel 文件名区分不了，交由调用方 --lang 指定
    return "cute_dsl" if files else "unknown"


def render_facts(detect, lang, grid, warps, routes):
    lines = []
    lines.append("语言 {0}".format(lang))

    clock = (detect or {}).get("clock") or {}
    hint = ((routes.get("existence", {}).get("device_timestamping", {})
             .get("hint_by_lang") or {}).get(lang))
    if clock.get("source"):
        lines.append("时钟 已在用 {0}（{1}）".format(clock["source"], clock.get("unit") or "?"))
        if not clock.get("cross_sm_alignable"):
            lines.append("     警告 此时钟不可跨 SM 对齐，必须换成 %globaltimer")
    else:
        lines.append("时钟 {0}".format(UNKNOWN))
    if hint:
        lines.append("     正确写法 {0}".format(hint))

    transport = (detect or {}).get("transport") or {}
    shifts = transport.get("tag_left_shifts")
    masks = transport.get("tag_masks")
    lines.append("位域左移量 {0}".format(shifts if shifts else UNKNOWN))
    if masks:
        lines.append("位域掩码 {0}".format(masks))
    lines.append("track 数 {0}".format(transport.get("num_tracks") or UNKNOWN))

    if grid and warps:
        lines.append("grid={0}，{1} warp/CTA，期望 {2} 条泳道".format(grid, warps, grid * warps))

    signals = (detect or {}).get("signals") or {}
    slot_files = signals.get("fixed_slot_files") or []
    lines.append("槽位公式所在文件 {0}".format(slot_files[0] if slot_files else UNKNOWN))

    abi = signals.get("abi_json_candidates") or []
    lines.append("格式说明文件 {0}".format(abi[0].get("file") if abi else UNKNOWN))

    decoders = signals.get("coupled_decoder_candidates") or []
    lines.append("解码器 {0}".format(decoders[0].get("file") if decoders else UNKNOWN))

    wiring = signals.get("host_wiring_files") or []
    if wiring:
        lines.append("host 接线 {0}".format(", ".join(wiring[:3])))

    sites = signals.get("instrumentation_call_sites")
    if sites:
        lines.append("已有打点文件数 {0}".format(sites))

    return "\n".join(lines)


def expand_gap(key, route, lang, index, total, message=None):
    out = ["[{0}/{1}] {2}".format(index, total, key)]
    if message:
        out.append("  现象 {0}".format(message))
    if route is None:
        out.append("  路由表里没有这一项，按判据原文自行定位，改完必须通过自验收")
        return "\n".join(out)
    if route.get("llm_owned"):
        out.append("  注意 这一项没有唯一正确答案，需要你判断")
    for line in route.get("discriminator", []):
        out.append("  判别 {0}".format(line))
    hint = (route.get("hint_by_lang") or {}).get(lang)
    if hint:
        out.append("  写法 {0}".format(hint))
    refs = [ref.replace("{lang}", lang) for ref in route.get("refs", [])]
    if refs:
        out.append("  参考 {0}".format("  ".join(refs)))
    for line in route.get("forbid", []):
        out.append("  禁止 {0}".format(line))
    return "\n".join(out)


def collect_gaps(detect, validate, routes, lang, batch):
    """返回 (stage, gap_text, batch_note)。存在性优先：没跑起来就谈不上验收。"""
    existence = routes.get("existence", {})
    validity = routes.get("validity", {})

    missing = list((detect or {}).get("missing") or [])
    if missing:
        batches = sorted({existence.get(name, {}).get("batch", 99) for name in missing})
        if batch is None:
            batch = batches[0]
        picked = [n for n in missing if existence.get(n, {}).get("batch", 99) == batch]
        skipped = [n for n in missing if n not in picked]
        blocks = [expand_gap(name, existence.get(name), lang, i + 1, len(picked))
                  for i, name in enumerate(picked)]
        note = ""
        if skipped:
            note = ("\n本轮只修上面 {0} 项。剩余 {1} 项按依赖顺序留到后续批次，"
                    "现在不要动：{2}").format(len(picked), len(skipped), ", ".join(skipped))
        return "存在性", "\n\n".join(blocks) + note, batch

    fails = list((validate or {}).get("fails") or [])
    if fails:
        blocks = [expand_gap(item.get("id", "?"), validity.get(item.get("id")), lang,
                             i + 1, len(fails), item.get("message"))
                  for i, item in enumerate(fails)]
        return "有效性", "\n\n".join(blocks), batch

    return None, None, batch


def render_loop(trace, grid, warps, phases, require_hw, hw_metric, rounds):
    validator = os.path.join(SKILL_ROOT, "4_validate", "validate_capability.py")
    cmd = ["  python3 -B {0} {1} \\".format(validator, trace or "<你的 trace.json.gz>"),
           "    --expect-blocks {0} --expect-warps {1} --expect-phases {2}".format(
               grid or "<grid>", warps or "<warps>", phases or "<phases>")]
    tail = " --require-task-layer"
    if require_hw:
        tail += " --require-hw"
        for metric in hw_metric:
            tail += ' --hw-required-metric "{0}"'.format(metric)
    cmd[-1] += tail
    return ("必须自己跑，FAIL 就继续修，最多 {0} 轮：\n".format(rounds)
            + "\n".join(cmd)
            + "\n  必须 PASS 才算完成。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", help="detect_instrumentation.py 的 verdict JSON")
    ap.add_argument("--validate", help="validate_capability.py --json 的输出")
    ap.add_argument("--project", help="项目根，默认取 detector 的 root")
    ap.add_argument("--lang", choices=sorted(set(LANG_BY_EXT.values()) | {"cute_dsl", "triton"}),
                    help="覆盖语言推断（.py 分不清 CuTe DSL 与 Triton）")
    ap.add_argument("--grid", type=int, help="CTA 数")
    ap.add_argument("--warps", type=int, help="每 CTA warp 数")
    ap.add_argument("--phases", type=int, help="期望 phase 数")
    ap.add_argument("--trace", help="自验收要检查的 trace 路径")
    ap.add_argument("--require-hw", action="store_true")
    ap.add_argument("--hw-required-metric", action="append", default=["SMs Active"])
    ap.add_argument("--gpu", default="0", help="限定使用的 GPU")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch", type=int, help="只渲染指定批次的存在性缺口")
    ap.add_argument("--skeleton", default=DEFAULT_SKELETON)
    ap.add_argument("--routes", default=DEFAULT_ROUTES)
    ap.add_argument("--out", help="写到文件，默认 stdout")
    args = ap.parse_args()

    if not args.detect and not args.validate:
        ap.error("至少要给 --detect 或 --validate")

    detect = load_json(args.detect) if args.detect else {}
    validate = load_json(args.validate) if args.validate else {}
    routes = load_json(args.routes)

    lang = guess_lang(detect, args.lang)
    grid = args.grid
    warps = args.warps
    phases = args.phases
    expect = validate.get("expect") or {}
    grid = grid or expect.get("blocks")
    warps = warps or expect.get("warps")
    phases = phases or expect.get("phases")

    stage, gaps, batch = collect_gaps(detect, validate, routes, lang, args.batch)
    if stage is None:
        print("无缺口：detector 的 missing 为空且 validator 没有 FAIL，不需要修补。",
              file=sys.stderr)
        return 0

    project = args.project or detect.get("root") or validate.get("trace") or "<项目>"
    require_hw = args.require_hw or bool(expect.get("hw"))
    hw_metric = expect.get("hw_metrics") or args.hw_required_metric

    with open(args.skeleton, "r", encoding="utf-8") as handle:
        text = handle.read()
    text = text.replace("{{PROJECT}}", str(project))
    text = text.replace("{{STAGE}}", stage)
    text = text.replace("{{FACTS}}", render_facts(detect, lang, grid, warps, routes))
    text = text.replace("{{GAPS}}", gaps)
    text = text.replace("{{LOOP}}", render_loop(
        args.trace or validate.get("trace"), grid, warps, phases,
        require_hw, hw_metric, args.rounds))
    text = text.replace("{{GPU_ID}}", str(args.gpu))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
        print("已写出 {0}（stage={1} batch={2} lang={3}）".format(
            args.out, stage, batch, lang), file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
