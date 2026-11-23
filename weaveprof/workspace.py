"""File-backed, per-round copies for build systems that cache by source pathname."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent.parent


def implementation(module):
    return importlib.import_module(module)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                       encoding="utf-8")
    os.replace(pending, path)


def digest(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _is_broad_home_or_root(directory: Path) -> bool:
    """Return True if `directory` is a filesystem root, home dir, or WSL /mnt/<drv>/Users/<name>."""
    res = directory.resolve()
    if res in (Path.home().resolve(), Path(res.anchor).resolve()):
        return True
    parts = [p.lower() for p in res.parts]
    # WSL mount root (/mnt/c), /mnt/c/Users, /mnt/c/Users/<user>, or /home/<user>
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        if len(parts) == 3:
            return True
        if len(parts) in (4, 5) and parts[3] == "users":
            return True
    if len(parts) == 3 and parts[0] == "/" and parts[1] in ("home", "users"):
        return True
    return False


def project_root(source):
    cwd = Path.cwd().resolve()
    candidates = (source.parent, *source.parent.parents)
    for directory in candidates:
        if _is_broad_home_or_root(directory):
            break
        if (directory / ".git").exists() or (directory / "pyproject.toml").is_file():
            return directory
    if source.is_relative_to(cwd) and not _is_broad_home_or_root(cwd):
        return cwd
    return source.parent


def cache_root(source):
    configured = os.environ.get("WEAVEPROF_CACHE")
    if configured:
        return Path(configured).resolve()
    if os.name != "nt":
        uid = os.getuid() if hasattr(os, "getuid") else 0
        return (Path(tempfile.gettempdir()) / f"weaveprof-cache-{uid}" / "probes").resolve()
    cwd = Path.cwd().resolve()
    if cwd.is_relative_to(project_root(source)):
        cwd = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "weaveprof"
    return cwd / ".weaveprof_cache"


def safe_products(source, *paths):
    files = [Path(p).resolve() for p in paths if p is not None]
    if source.resolve() in files or len(files) != len(set(files)):
        raise ValueError("Source, probe, feedback and output must have distinct paths")


class ShadowWorkspace:
    def __init__(self, source, location=None):
        self.source = Path(source).resolve(strict=True)
        self.project = project_root(self.source)
        default_base = (
            Path(tempfile.gettempdir()) / f"weaveprof-runs-{os.getuid() if hasattr(os, 'getuid') else 0}"
            if os.name != "nt"
            else Path.home() / ".cache" / "weaveprof" / "runs"
        )
        base = Path(location or os.environ.get("WEAVEPROF_WORKDIR") or default_base).resolve()
        if base.is_relative_to(self.project):
            raise ValueError(f"Shadow workspace must be outside {self.project}")
        self.path = base / uuid.uuid4().hex
        self.path.mkdir(parents=True)
        self.sha = digest(self.source)
        self.l0_cache = self.path / "_l0_cache"
        self.l0_cache.mkdir(exist_ok=True)

    def prepare(self, number, spec):
        directory = self.path / f"round-{number}"
        directory.mkdir()
        original = directory / "L0"
        instrumented = directory / "L_inst"
        ignore = shutil.ignore_patterns(
            ".git", ".weaveprof_cache", "__pycache__", ".venv", "venv", "env", ".pytest_cache", "*.pyc",
            "*.perfetto-trace", "*.perfetto*", "*.gz", "*.sqlite", "*.nsys-rep",
            "*.fi02.bin", "*.wp02.bin", "*.out.bin", "*.cubin", "*.fatbin", "*.pt", "*.pth",
            "build", "dist", "third_party", "node_modules", ".claude", ".codex", ".qwen", ".agents", ".gemini", ".cache",
        )
        shutil.copytree(self.project, original, ignore=ignore)
        relative = self.source.relative_to(self.project)
        if digest(original / relative) != self.sha:
            raise ValueError("Source changed since probe identification")
        shutil.copytree(original, instrumented)
        source_text = (original / relative).read_text(encoding="utf-8")
        engine = implementation("weaveprof.pipeline.instrument.rule_engine")
        lowered = engine.inject_probes(source_text, spec, check_sha=False)
        if not lowered["ok"]:
            raise ValueError(str(lowered["diagnostics"]))
        os.chmod(instrumented / relative, 0o666)
        (instrumented / relative).write_text(lowered["instrumented_source"], encoding="utf-8")
        write_json(directory / "names.json", lowered["names"])
        write_json(directory / "probe.json", spec)
        return directory, original / relative, instrumented / relative

    def environment(self, directory, baseline, instrumented):
        tree = directory / "L_inst"
        libs = ROOT / "recorders"
        env = dict(os.environ)
        env.pop("WEAVEPROF_SUPERVISOR_KEY_FILE", None)
        env.pop("WEAVEPROF_SUPERVISOR_HMAC_KEY", None)
        env.update(WEAVEPROF_ROOT=str(ROOT), WEAVEPROF_BASELINE_SOURCE=str(baseline),
                   WEAVEPROF_INSTRUMENTED_SOURCE=str(instrumented), WEAVEPROF_NAMES=str(directory / "names.json"),
                   WEAVEPROF_ROUND_DIR=str(directory), WEAVEPROF_L0_CACHE_DIR=str(self.l0_cache),
                   PYTHONDONTWRITEBYTECODE="1",
                   TORCH_EXTENSIONS_DIR=str(directory / "extensions"), TRITON_CACHE_DIR=str(directory / "jit"))
        env["PYTHONPATH"] = os.pathsep.join(map(str, [tree, ROOT, libs / "triton", libs / "cutedsl",
            *[p for p in sys.path if p and not Path(p).resolve().is_relative_to(self.project)]]))
        env["CPATH"] = os.pathsep.join(map(str, [libs / "cuda_cpp", libs / "hip_cpp",
                                                *filter(None, env.get("CPATH", "").split(os.pathsep))]))
        return env

    def command(self, bench, directory, *, baseline=False):
        tree = directory / ("L0" if baseline else "L_inst")
        cwd = Path.cwd().resolve()
        mapped_cwd = tree / cwd.relative_to(self.project) if cwd.is_relative_to(self.project) else tree
        argv = []
        for token in bench:
            path = Path(token)
            if not token.startswith("-") and path.exists():
                path = path.resolve()
                if path.is_relative_to(self.project):
                    token = str(tree / path.relative_to(self.project))
                elif path.is_file() and path.suffix == ".py":
                    target = tree / path.name
                    if target.exists() and digest(path) != digest(target):
                        raise ValueError(f"Conflicting external benchmark name: {target}")
                    shutil.copy2(path, target)
                    token = str(target)
            argv.append(token)
        return argv, mapped_cwd
