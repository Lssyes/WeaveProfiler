#!/usr/bin/env python3
"""Offline structure/compile checks plus MI308X runtime checks for fi_lite_hip.h."""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "recorders" / "hip_cpp" / "fi_lite_hip.h"


HARNESS = r'''
#include "fi_lite_hip.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static void check_impl(bool ok, const char* expr, int line) {
    if (ok) return;
    std::cerr << "CHECK failed at line " << line << ": " << expr << "\n";
    std::exit(2);
}
#define CHECK(expr) check_impl((expr), #expr, __LINE__)

__global__ void probe_kernel(uint32_t* lanes, uint32_t role_base) {
    const uint64_t linear = fi_linear_tid();
    if (lanes != nullptr) {
        const uint64_t block_threads =
            (uint64_t)blockDim.x * (uint64_t)blockDim.y * (uint64_t)blockDim.z;
        const uint64_t index = (uint64_t)blockIdx.x * block_threads + linear;
        lanes[index] = fi_lane();
    }
    FI_OPEN(role_base + blockIdx.x);
    FI_B(1u, 0x12345678u);
    FI_E(1u, 0x87654321u);
    FI_CLOSE();
}

static std::string join_path(const std::string& dir, const char* name) {
    return dir + (dir.empty() || dir.back() == '/' ? "" : "/") + name;
}

static bool has_arg(int argc, char** argv, const char* wanted) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], wanted) == 0) return true;
    return false;
}

static void host_checks() {
    FiBuf b{};
    CHECK(fi_alloc(&b, 0u, 1u, 1u) == hipErrorInvalidValue);
    CHECK(fi_alloc(&b, 1u, 1u, 0u) == hipErrorInvalidValue);
#if FI_GRAN == 0
    CHECK(fi_alloc(&b, UINT32_MAX, 2u, 16u) == hipErrorInvalidValue);
#endif
    // (2^32-1) records * 16 bytes overflows size_t when multiplied by 16.
    CHECK(fi_alloc(&b, UINT32_MAX, 1u, UINT32_MAX) == hipErrorInvalidValue);
    size_t count = 0;
    CHECK(!fi_record_count(0u, 1u, &count));
    CHECK(!fi_record_count(1u, 0u, &count));
    CHECK(!fi_record_bytes(UINT32_MAX, UINT32_MAX, &count));
    FiBuf empty{};
    CHECK(fi_reset(&empty, 0) == hipErrorInvalidValue);

    FiBuf meta_buf{};
    meta_buf.timer_hz = 12345u;
    std::snprintf(meta_buf.arch, sizeof(meta_buf.arch), "%s", "gfx942");
    const char* metas[] = {
        "{}",
        "{ }",
        "{\"owner\": 1}",
        " {\n \"owner\": 1, \"nested\": {\"ok\": true}\n} \n",
        "not-json",
    };
    for (const char* source : metas) {
        const std::string augmented = fi_meta_augment(source, &meta_buf);
        std::cout << "META_JSON:\"" << fi_json_escape(augmented.c_str()) << "\"\n";
    }
}

static void check_lanes(uint32_t* host_lanes, uint32_t n_ctas, uint32_t block_threads) {
    for (uint32_t cta = 0; cta < n_ctas; ++cta)
        for (uint32_t i = 0; i < block_threads; ++i)
            CHECK(host_lanes[(size_t)cta * block_threads + i] == (i & 63u));
}

int main(int argc, char** argv) {
    host_checks();
    if (has_arg(argc, argv, "--host-only")) {
        std::cout << "HOST_ONLY_PASS\n";
        return 0;
    }
    if (argc < 2) {
        std::cerr << "usage: test_hip_recorder <output-dir> [--require-runtime]\n";
        return 2;
    }
    const std::string out_dir = argv[1];
    int device_count = 0;
    hipError_t e = hipGetDeviceCount(&device_count);
    if (e != hipSuccess || device_count <= 0) {
        std::cout << "DEVICE_RUNTIME_UNVERIFIED reason="
                  << (e == hipSuccess ? "no-device" : hipGetErrorString(e)) << "\n";
        return has_arg(argc, argv, "--require-runtime") ? 3 : 0;
    }
    int chosen = device_count > 1 ? 1 : 0;
    if (const char* env = std::getenv("FI_HIP_TEST_DEVICE_INDEX")) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed >= 0 && parsed < device_count) chosen = (int)parsed;
    }
    CHECK(hipSetDevice(chosen) == hipSuccess);
    int current = -1;
    CHECK(hipGetDevice(&current) == hipSuccess);
    CHECK(current == chosen);
    hipDeviceProp_t prop{};
    CHECK(hipGetDeviceProperties(&prop, chosen) == hipSuccess);
    std::string want_arch(prop.gcnArchName);
    const size_t colon = want_arch.find(':');
    if (colon != std::string::npos) want_arch.resize(colon);

#if FI_GRAN == 0
    const uint32_t n_ctas = 1u;
    const uint32_t warps_arg = 2u;
    const dim3 block(32u, 2u, 2u);
    const uint32_t block_threads = 128u;
    const uint32_t want_slots = 2u;
    const uint32_t want_wpc = 2u;
    const uint32_t want_gran = 0u;
#else
    const uint32_t n_ctas = 2u;
    const uint32_t warps_arg = 99u;  // fi_alloc must force this to 1
    const dim3 block(2u, 3u, 4u);
    const uint32_t block_threads = 24u;
    const uint32_t want_slots = 2u;
    const uint32_t want_wpc = 1u;
    const uint32_t want_gran = 1u;
#endif
    const uint32_t cap = 8u;
    FiBuf b{};
    CHECK(fi_alloc(&b, n_ctas, warps_arg, cap) == hipSuccess);
    CHECK(b.n_slots == want_slots);
    CHECK(b.warps_per_cta == want_wpc);
    CHECK(b.arch[0] != '\0');
    CHECK(want_arch.empty() || want_arch == b.arch);
    CHECK(fi_attach(&b) == hipSuccess);

    const size_t lane_count = (size_t)n_ctas * block_threads;
    uint32_t* d_lanes = nullptr;
    CHECK(hipMalloc(&d_lanes, lane_count * sizeof(uint32_t)) == hipSuccess);
    CHECK(hipMemset(d_lanes, 0xff, lane_count * sizeof(uint32_t)) == hipSuccess);
    probe_kernel<<<n_ctas, block, 0, 0>>>(d_lanes, 0u);
    CHECK(hipGetLastError() == hipSuccess);
    CHECK(hipDeviceSynchronize() == hipSuccess);
    std::vector<uint32_t> lanes(lane_count, 0u);
    CHECK(hipMemcpy(lanes.data(), d_lanes, lane_count * sizeof(uint32_t),
                    hipMemcpyDeviceToHost) == hipSuccess);
    check_lanes(lanes.data(), n_ctas, block_threads);
    CHECK(hipFree(d_lanes) == hipSuccess);

    const std::string main_dump = join_path(out_dir, "main.fi.bin");
    CHECK(fi_dump(&b, main_dump.c_str(), "{}") == 0);
    const char* metas[] = {
        "{}",
        "{ }",
        "{\"owner\": \"hip-test\", \"nested\": {\"ok\": true}}",
        " \n { \"owner\": \"hip-test\" } \n ",
    };
    std::vector<std::string> meta_paths;
    for (size_t i = 0; i < sizeof(metas) / sizeof(metas[0]); ++i) {
        const std::string path = join_path(out_dir, ("meta_" + std::to_string(i) + ".fi.bin").c_str());
        CHECK(fi_dump(&b, path.c_str(), metas[i]) == 0);
        meta_paths.push_back(path);
    }

    // A host view with fewer slots must reject CTA 1 before touching the second slot.
    FiBuf slot_b{};
    CHECK(fi_alloc(&slot_b, 2u, 1u, 4u) == hipSuccess);
    const size_t slot_sentinel_bytes = 2u * 4u * sizeof(FiRec);
    CHECK(hipMemset(slot_b.rec, 0xa5, slot_sentinel_bytes) == hipSuccess);
    slot_b.n_slots = 1u;
    CHECK(fi_attach(&slot_b) == hipSuccess);
    probe_kernel<<<2u, 24u, 0, 0>>>(nullptr, 100u);
    CHECK(hipGetLastError() == hipSuccess);
    CHECK(hipDeviceSynchronize() == hipSuccess);
    uint32_t slot_overflow = 0u;
    CHECK(hipMemcpy(&slot_overflow, slot_b.overflow, sizeof(slot_overflow),
                    hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(slot_overflow == 2u);
    std::vector<unsigned char> slot_raw(slot_sentinel_bytes, 0u);
    CHECK(hipMemcpy(slot_raw.data(), slot_b.rec, slot_raw.size(),
                    hipMemcpyDeviceToHost) == hipSuccess);
    for (size_t i = 4u * sizeof(FiRec); i < slot_raw.size(); ++i)
        CHECK(slot_raw[i] == 0xa5u);
    const FiRec* slot0 = reinterpret_cast<const FiRec*>(slot_raw.data());
    CHECK((slot0[0].tag & FI_TAG_OPEN) != 0u);

    // A host view with cap=1 must stop after one record and leave the rest untouched.
    FiBuf cap_b{};
    CHECK(fi_alloc(&cap_b, 1u, 1u, 4u) == hipSuccess);
    const size_t cap_sentinel_bytes = 4u * sizeof(FiRec);
    CHECK(hipMemset(cap_b.rec, 0x5a, cap_sentinel_bytes) == hipSuccess);
    cap_b.cap = 1u;
    CHECK(fi_attach(&cap_b) == hipSuccess);
    probe_kernel<<<1u, 24u, 0, 0>>>(nullptr, 200u);
    CHECK(hipGetLastError() == hipSuccess);
    CHECK(hipDeviceSynchronize() == hipSuccess);
    uint32_t cap_overflow = 0u;
    CHECK(hipMemcpy(&cap_overflow, cap_b.overflow, sizeof(cap_overflow),
                    hipMemcpyDeviceToHost) == hipSuccess);
    CHECK(cap_overflow == 1u);
    std::vector<unsigned char> cap_raw(cap_sentinel_bytes, 0u);
    CHECK(hipMemcpy(cap_raw.data(), cap_b.rec, cap_raw.size(),
                    hipMemcpyDeviceToHost) == hipSuccess);
    const FiRec* cap_rec = reinterpret_cast<const FiRec*>(cap_raw.data());
    CHECK((cap_rec[0].tag & FI_TAG_OPEN) != 0u);
    for (size_t i = sizeof(FiRec); i < cap_raw.size(); ++i)
        CHECK(cap_raw[i] == 0x5au);

    std::cout << "{\"status\":\"runtime-pass\""
              << ",\"main_dump\":\"" << main_dump << "\""
              << ",\"meta_dumps\":[";
    for (size_t i = 0; i < meta_paths.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << '"' << meta_paths[i] << '"';
    }
    std::cout << "],\"want_slots\":" << want_slots
              << ",\"want_wpc\":" << want_wpc
              << ",\"want_gran\":" << want_gran
              << ",\"cap\":" << cap
              << ",\"chosen_device\":" << chosen
              << ",\"device_count\":" << device_count
              << ",\"arch\":\"" << b.arch << "\""
              << ",\"slot_overflow\":" << slot_overflow
              << ",\"cap_overflow\":" << cap_overflow
              << "}\n";

    fi_free(&slot_b);
    fi_free(&cap_b);
    fi_free(&b);
    return 0;
}
'''


def fail(message: str) -> "None":
    raise SystemExit(f"HIP RECORDER FAIL: {message}")


def structure_checks() -> None:
    source = HEADER.read_text(encoding="utf-8")
    required = [
        "fi_linear_tid()",
        "hipGetDevice(&device)",
        "hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, device)",
        "hipGetDeviceProperties(&prop, device)",
        "b->warps_per_cta = 1u;",
        "fi_record_bytes(slots, cap, &all_bytes)",
        "fi_record_count(b->n_slots, b->cap, &record_count)",
        "std::string(empty_obj ? \"\" : \",\") + inject",
    ]
    for marker in required:
        if marker not in source:
            fail(f"missing required recorder structure: {marker}")
    if "hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0)" in source:
        fail("wall-clock attribute still queries hard-coded device 0")
    if "hipGetDeviceProperties(&prop, 0)" in source:
        fail("device properties still query hard-coded device 0")
    if not re.search(r"fi_u64_mul_add\(\(uint64_t\)blockDim\.x,\s*row,\s*\(uint64_t\)threadIdx\.x", source):
        fail("wave lane is not derived from the full linear thread id")
    if source.count("fi_record_bytes(") < 3:
        fail("all allocation/reset/dump byte-size paths must use the checked helper")


def compile_harness(hipcc: str, build_dir: Path, gran: int) -> Path:
    source = build_dir / f"hip_recorder_gran{gran}.cpp"
    binary = build_dir / f"hip_recorder_gran{gran}"
    source.write_text(HARNESS, encoding="utf-8")
    cmd = [
        hipcc,
        "-std=c++17",
        "-O2",
        "--offload-arch=gfx942",
        "-DFI_LEVEL=2",
        f"-DFI_GRAN={gran}",
        f"-I{ROOT / 'recorders' / 'hip_cpp'}",
        str(source),
        "-o",
        str(binary),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        fail(f"hipcc compile failed for FI_GRAN={gran}")
    return binary


def run_host_only(binary: Path, gran: int) -> None:
    proc = subprocess.run(
        [str(binary), "--host-only"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        fail(f"host-only executable failed for FI_GRAN={gran}")
    if "HOST_ONLY_PASS" not in proc.stdout:
        fail(f"host-only harness did not report PASS for FI_GRAN={gran}")
    encoded = [line.split("META_JSON:", 1)[1] for line in proc.stdout.splitlines()
               if line.startswith("META_JSON:")]
    metas = [json.loads(text) for text in encoded]
    if len(metas) != 5:
        fail(f"expected five augmented meta documents, got {len(metas)}")
    for i, text in enumerate(metas):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            fail(f"gran={gran} meta variant {i} is invalid JSON: {exc}: {text!r}")
        if not isinstance(obj, dict):
            fail(f"gran={gran} meta variant {i} is not an object")
        if obj.get("timer_hz") != 12345 or obj.get("gran") != gran or obj.get("arch") != "gfx942":
            fail(f"gran={gran} meta variant {i} lost injected fields: {obj!r}")


def decode_dump(path: Path) -> tuple[dict, list[int], list[list[tuple[int, int, int]]]]:
    data = path.read_bytes()
    if len(data) < 32 or data[:8] != b"FILITE02":
        fail(f"bad FILITE02 header: {path}")
    n_slots, cap, wpc, overflow, timer, meta_len = struct.unpack_from("<6I", data, 8)
    meta_end = 32 + meta_len
    if meta_end > len(data):
        fail(f"truncated metadata: {path}")
    try:
        meta = json.loads(data[32:meta_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid metadata JSON in {path}: {exc}")
    if meta_end + 4 * n_slots > len(data):
        fail(f"truncated count table: {path}")
    counts = list(struct.unpack_from(f"<{n_slots}I", data, meta_end))
    offset = meta_end + 4 * n_slots
    slots: list[list[tuple[int, int, int]]] = []
    for count in counts:
        records = []
        for _ in range(count):
            if offset + 16 > len(data):
                fail(f"truncated record body: {path}")
            records.append(struct.unpack_from("<QII", data, offset))
            offset += 16
        slots.append(records)
    if offset != len(data):
        fail(f"unexpected trailing bytes in dump: {path}")
    header = {
        "n_slots": n_slots,
        "cap": cap,
        "warps_per_cta": wpc,
        "overflow": overflow,
        "timer": timer,
        "meta": meta,
    }
    return header, counts, slots


def check_event_shape(counts: list[int], slots: list[list[tuple[int, int, int]]], cap: int) -> None:
    if counts != [4, 4]:
        fail(f"expected four records in each of two slots, got {counts}")
    for slot in slots:
        tags = [record[1] for record in slot]
        if tags[0] & 0x40000000 == 0:
            fail(f"slot does not start with OPEN: {tags}")
        if tags[1] != 1:
            fail(f"FI_B record changed: {tags}")
        if tags[2] != (0x80000000 | 1):
            fail(f"FI_E record changed: {tags}")
        if tags[3] != 0x08000000:
            fail(f"slot does not end with CLOSE: {tags}")
    if not all(count <= cap for count in counts):
        fail(f"record count exceeds cap: counts={counts}, cap={cap}")


def runtime_checks(binaries: list[tuple[int, Path]]) -> None:
    for gran, binary in binaries:
        with tempfile.TemporaryDirectory(prefix=f"fi_hip_gran{gran}_") as out:
            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            env["FI_TIMER_HZ"] = ""  # empty override must fall back to the current device's clock
            proc = subprocess.run(
                [str(binary), out, "--require-runtime"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
            if proc.returncode != 0:
                print(proc.stdout, file=sys.stderr)
                fail(f"runtime harness failed for FI_GRAN={gran}")
            print(proc.stdout.strip())
            lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
            if not lines:
                fail(f"runtime harness printed no summary for FI_GRAN={gran}")
            summary = json.loads(lines[-1])
            if summary.get("status") != "runtime-pass":
                fail(f"bad runtime status for FI_GRAN={gran}: {summary}")
            if summary["device_count"] > 1 and summary["chosen_device"] != 1:
                fail("multi-GPU runtime did not execute on non-default device 1")
            if summary["arch"] != "gfx942":
                fail(f"expected MI308X gfx942, got {summary['arch']}")
            main_header, counts, slots = decode_dump(Path(summary["main_dump"]))
            if main_header["n_slots"] != summary["want_slots"]:
                fail(f"wrong n_slots for FI_GRAN={gran}: {main_header}")
            if main_header["warps_per_cta"] != summary["want_wpc"]:
                fail(f"wrong warps_per_cta for FI_GRAN={gran}: {main_header}")
            if main_header["overflow"] != 0:
                fail(f"unexpected overflow in normal run for FI_GRAN={gran}: {main_header}")
            meta = main_header["meta"]
            if meta.get("gran") != summary["want_gran"]:
                fail(f"wrong gran in dump metadata for FI_GRAN={gran}: {meta}")
            if meta.get("arch") != "gfx942" or int(meta.get("timer_hz", 0)) <= 0:
                fail(f"missing timer/arch metadata for FI_GRAN={gran}: {meta}")
            check_event_shape(counts, slots, summary["cap"])
            for path in summary["meta_dumps"]:
                meta_header, _, _ = decode_dump(Path(path))
                if meta_header["meta"].get("gran") != summary["want_gran"]:
                    fail(f"meta variant lost gran for FI_GRAN={gran}: {path}")
            if summary["slot_overflow"] != 2:
                fail(f"slot overflow bit not set for FI_GRAN={gran}")
            if summary["cap_overflow"] != 1:
                fail(f"capacity overflow bit not set for FI_GRAN={gran}")


def main() -> int:
    compile_only = "--compile-only" in sys.argv[1:]
    require_runtime = "--require-runtime" in sys.argv[1:]
    structure_checks()
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        print("HIP RECORDER STRUCTURE PASS")
        print("HIP RECORDER COMPILE/RUNTIME UNVERIFIED: hipcc not found")
        return 2 if require_runtime else 0
    with tempfile.TemporaryDirectory(prefix="fi_hip_build_") as build:
        build_dir = Path(build)
        binaries = [
            (0, compile_harness(hipcc, build_dir, 0)),
            (1, compile_harness(hipcc, build_dir, 1)),
        ]
        for gran, binary in binaries:
            run_host_only(binary, gran)
        if compile_only:
            print("HIP RECORDER STRUCTURE/HOST PASS: 2 builds compiled; device runtime unverified")
            return 0
        runtime_checks(binaries)
    print("HIP RECORDER PASS: structure, 2 hipcc builds, MI308X runtime, JSON, overflow sentinels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
