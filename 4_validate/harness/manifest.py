#!/usr/bin/env python3
"""Content-addressed manifest lock for the weaveprofiler skill tree.

build:   hash every skill file and write 4_validate/harness/MANIFEST.json.
verify:  re-hash the working tree and compare it with MANIFEST.json.  The
         default anchor is the checked-in tree manifest, so an untracked but
         complete skill snapshot can be validated; git remains available for
         callers that want a commit-bound check.

Exit codes: 0 clean | 1 differences | 2 usage/internal | 3 anchor unavailable.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "skill_manifest.v1"
EXCLUDE_DIRS = {"__pycache__", "4_validate/harness/state", "4_validate/harness/testdata"}
MANIFEST_REL = Path("4_validate") / "harness" / "MANIFEST.json"
EXCLUDE_FILES = {MANIFEST_REL.as_posix()}
EXCLUDE_SUFFIXES = (".pyc",)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_skill_files(root):
    """Yield POSIX relative paths of every hashed file, sorted."""
    root = Path(root).resolve()
    paths = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        dirnames.sort()
        rel_dir = os.path.relpath(dirpath, str(root))
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git")
                       and (rel_dir + "/" + d if rel_dir else d) not in EXCLUDE_DIRS
                       and not (rel_dir == "harness" and d in ("state", "testdata"))]
        for name in sorted(filenames):
            rel = (rel_dir + "/" + name) if rel_dir else name
            if rel in EXCLUDE_FILES or name.endswith(EXCLUDE_SUFFIXES):
                continue
            if name == "MANIFEST.json" and (rel_dir == "harness" or rel_dir.endswith("/harness")):
                continue
            paths.append(rel)
    return sorted(paths)


def root_hash(files):
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[rel].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build(root):
    root = Path(root).resolve()
    files = {rel: sha256_file(root / rel) for rel in iter_skill_files(root)}
    payload = {
        "schema_version": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        # Keep the checked-in anchor portable across cards and hosts.  The
        # verifier already receives the real root as an explicit argument.
        "skill_root": ".",
        "file_count": len(files),
        "root_sha256": root_hash(files),
        "files": files,
    }
    out = root / MANIFEST_REL
    with open(str(out), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def committed_manifest(root):
    """Return (manifest_dict, error). Anchor = git object store, not the tree."""
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=15)
        if top.returncode != 0:
            return None, "git rev-parse failed: %s" % top.stderr.strip()
        toplevel = Path(top.stdout.strip()).resolve()
        rel = (Path(root).resolve() / MANIFEST_REL).relative_to(toplevel)
        rel = str(rel).replace(os.sep, "/")
        show = subprocess.run(["git", "-C", str(toplevel), "show", "HEAD:" + rel],
                              capture_output=True, text=True, timeout=15)
        if show.returncode != 0:
            return None, ("MANIFEST.json not in git HEAD at %s -- commit the skill "
                          "tree first" % rel)
        return json.loads(show.stdout), None
    except (OSError, ValueError) as error:
        return None, "git anchor unavailable: %s" % error


def diff_against(files_expected, root):
    changed, missing = [], []
    for rel, want in files_expected.items():
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif sha256_file(path) != want:
            changed.append(rel)
    added = [rel for rel in iter_skill_files(root) if rel not in files_expected]
    return changed, missing, added


def verify(root, anchor):
    root = Path(root).resolve()
    report = {"status": "CLEAN", "anchor": anchor, "skill_root": str(root)}
    if anchor == "git":
        manifest, error = committed_manifest(root)
        if error:
            report.update({"status": "ANCHOR_UNAVAILABLE", "reason": error})
            return report, 3
    else:
        manifest_path = root / MANIFEST_REL
        if not manifest_path.is_file():
            report.update({"status": "ANCHOR_UNAVAILABLE",
                           "reason": "no working-tree MANIFEST.json"})
            return report, 3
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA:
        report.update({"status": "DIFF", "reason": "manifest schema mismatch"})
        return report, 1
    changed, missing, added = diff_against(manifest.get("files", {}), root)
    report["root_sha256"] = manifest.get("root_sha256")
    report["file_count"] = manifest.get("file_count")
    if changed or missing or added:
        report.update({"status": "DIFF", "changed": changed,
                       "missing": missing, "added": added})
        return report, 1
    return report, 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--skill-root", default=str(Path(__file__).resolve().parent.parent.parent))
    parser.add_argument("--anchor", choices=("git", "tree"), default="tree")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "build":
        payload = build(args.skill_root)
        print("MANIFEST=%s" % (Path(args.skill_root) / MANIFEST_REL))
        print("ROOT_SHA256=%s FILES=%d" % (payload["root_sha256"], payload["file_count"]))
        return 0
    report, code = verify(args.skill_root, args.anchor)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print("MANIFEST_STATUS=%s ANCHOR=%s" % (report["status"], report["anchor"]))
        for key in ("changed", "missing", "added"):
            for rel in report.get(key, []):
                print("  %s: %s" % (key, rel))
        if report.get("reason"):
            print("REASON=%s" % report["reason"])
    return code


if __name__ == "__main__":
    sys.exit(main())
