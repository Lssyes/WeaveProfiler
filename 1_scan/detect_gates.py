#!/usr/bin/env python3
"""Probe a repository for gate scripts and emit a machine-readable write-zone map."""

from __future__ import annotations

import argparse
import ast
import json
import os
import posixpath
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = "gate-constraints/v1"

DEFAULT_EXCLUDES = (
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".work",
    ".selftest",
    "build",
    "node_modules",
    "profile/refs",
    "profile/scripts",
)

MAX_FILE_BYTES = 2 * 1024 * 1024

GATE_DIR_RE = re.compile(r"^\.?(?:verif\w*|gates?|validat\w*|check\w*|ci|cicd|githooks)$")
GATE_NAME_RES = (
    (re.compile(r"^.*_gate\.py$"), "filename ~ *_gate.py"),
    (re.compile(r"^gate_.*\.py$"), "filename ~ gate_*.py"),
    (re.compile(r"^validate_.*\.py$"), "filename ~ validate_*.py"),
    (re.compile(r"^check_.*\.py$"), "filename ~ check_*.py"),
    (re.compile(r"^.*_equivalence\.py$"), "filename ~ *_equivalence.py"),
    (re.compile(r"^test_.*\.py$"), "filename ~ test_*.py"),
    (re.compile(r"^.*_test\.py$"), "filename ~ *_test.py"),
    (re.compile(r"^conftest\.py$"), "filename ~ conftest.py"),
)

CONST_PREFIX_RE = re.compile(
    r"^(ALLOWED|ALLOWLIST|REQUIRED|FORBIDDEN|OPTIONAL|EXPECTED|IMMUTABLE|DENIED|PERMITTED)"
    r"(?:_[A-Z0-9_]*)?$"
)
CONST_SUFFIX_RE = re.compile(r"^[A-Z0-9_]*_(WHITELIST|BLACKLIST|ALLOWLIST|DENYLIST)$")

KIND_BY_TOKEN = {
    "ALLOWED": "allow",
    "ALLOWLIST": "allow",
    "PERMITTED": "allow",
    "WHITELIST": "allow",
    "REQUIRED": "allow",
    "OPTIONAL": "allow",
    "EXPECTED": "allow",
    "FORBIDDEN": "deny",
    "DENIED": "deny",
    "BLACKLIST": "deny",
    "DENYLIST": "deny",
    "IMMUTABLE": "immutable",
}

ARG_NAMESPACES = ("args", "options", "opts", "ns", "parsed", "namespace")
LISTING_METHODS = ("iterdir", "glob", "rglob")
WRAPPERS = ("sorted", "set", "list", "tuple", "frozenset")
PATH_FACTORIES = ("Path", "PurePath", "PosixPath", "PurePosixPath")

FILE_EXT_HINTS = (
    ".py", ".cu", ".cuh", ".cpp", ".cc", ".hpp", ".h", ".json", ".md", ".txt",
    ".log", ".cubin", ".ptx", ".sass", ".so", ".yaml", ".yml", ".toml", ".cfg",
    ".ini", ".csv", ".sh", ".lock", ".gitignore",
)
TREE_HINT_COMPONENTS = {
    "include", "includes", "inc", "src", "csrc", "source", "sources",
    "headers", "kernels",
}
DUMP_TOOL_RE = re.compile(r"cuobjdump|nvdisasm|--dump-sass|objdump|llvm-objdump|readelf")
BINARY_TOKEN_RE = re.compile(r"cubin|sass|ptx|elf|binary|objdump|disasm", re.IGNORECASE)

CONTRACT_FILENAMES = ("AGENTS.md", "README.md", "CONTRIBUTING.md", "CONTRIBUTING.rst")

TEXT_CATEGORY_RULES = (
    ("artifact_location", (
        r"/tmp\b", r"\.work/", r"`\.work`", r"temporary artifact", r"temp(?:orary)? file",
        r"scratch (?:dir|space)", r"临时产物", r"临时文件", r"临时目录", r"temporary", r"临时",
    )),
    ("authorization_required", (
        r"requires? authorization", r"explicit authorization", r"explicit approval",
        r"explicit workflow", r"explicit permission", r"without explicit", r"needs? approval",
        r"requires? approval", r"requires? review", r"需授权", r"需要授权", r"需经授权",
        r"需要审核", r"需审核", r"需经批准", r"授权",
    )),
    ("immutable_path", (
        r"immutable", r"read-?only", r"do not modify", r"don't modify",
        r"must not (?:be )?(?:modif|chang|edit|touch)", r"never (?:modify|change|edit|touch)",
        r"do not (?:edit|change|touch)", r"不可修改", r"不要修改", r"禁止修改", r"不得修改",
        r"禁止改动", r"不可变",
    )),
    ("count_constraint", (
        r"exactly", r"\bexact\b", r"add no extra", r"no extra files", r"must not exceed",
        r"at most", r"exceeds \d+", r"恰好", r"正好", r"至多", r"最多", r"不得超过",
    )),
)

GENERIC_TEXT_KEYWORDS = (
    r"immutable", r"do not modify", r"must not", r"never\b", r"\bonly\b", r"\bexactly\b",
    r"/tmp\b", r"temporary", r"requires? authorization", r"不可修改", r"不要修改", r"禁止",
    r"仅", r"恰好", r"临时", r"需授权", r"需要审核",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+")
BACKTICK_RE = re.compile(r"`([^`]+)`")
BARE_TMP_RE = re.compile(r"(?<![\w`/])(/tmp(?:/[\w.\-]*)*)")
BLOCK_BREAK_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}\s|\|)")

DIFF_ALLOW_RE = re.compile(
    r"^(?:allowed|allow|allowlist|allow_list|permitted|whitelist|white_list)\w*$"
)
DIFF_CHANGE_RE = re.compile(
    r"^(?:changed|changes|change|diff|diffs|modified|delta|dirty|touched|edited|edits)\w*$"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm_path(text):
    if not text:
        return text
    if text.startswith("/"):
        return posixpath.normpath(text)
    return posixpath.normpath(text) if "/" in text else text


def join_path(base, part):
    if part is None:
        return None
    if base is None:
        return norm_path(part)
    if part.startswith("/"):
        return norm_path(part)
    return norm_path(base.rstrip("/") + "/" + part)


def parent_path(text):
    if not text:
        return None
    return str(PurePosixPath(text).parent)


def is_templated(text):
    return bool(text) and ("<" in text or "{" in text)


def has_file_ext(text):
    lowered = text.lower()
    return any(lowered.endswith(ext) for ext in FILE_EXT_HINTS)


def build_excludes(user_values, keep_defaults):
    values = list(DEFAULT_EXCLUDES) if keep_defaults else []
    for item in user_values or []:
        for chunk in item.split(","):
            chunk = chunk.strip().strip("/")
            if chunk and chunk not in values:
                values.append(chunk)
    return values


def make_exclusion_test(excludes):
    bare = {item for item in excludes if "/" not in item}
    prefixes = tuple(item for item in excludes if "/" in item)

    def excluded(rel):
        if any(part in bare for part in rel.split("/")):
            return True
        return any(rel == prefix or rel.startswith(prefix + "/") for prefix in prefixes)

    return excluded


def walk_repo(root, excludes):
    excluded = make_exclusion_test(excludes)
    files = []
    skipped = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        keep = []
        for name in sorted(dirnames):
            child = f"{rel_dir}/{name}" if rel_dir else name
            if excluded(child):
                skipped.append(child + "/")
            else:
                keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if excluded(rel):
                continue
            path = Path(dirpath) / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(path)
    return files, sorted(set(skipped))


def read_text(path):
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    if b"\0" in blob[:8192]:
        return None
    return blob.decode("utf-8", errors="replace")


def gate_reasons(root, path):
    rel = path.relative_to(root).as_posix()
    reasons = []
    for component in rel.split("/")[:-1]:
        if GATE_DIR_RE.match(component):
            reasons.append(f"path component '{component}'")
    for pattern, label in GATE_NAME_RES:
        if pattern.match(path.name):
            reasons.append(label)
    return reasons


class ModuleAnalysis:
    def __init__(self, path, root, source):
        self.path = path
        self.root = root
        self.root_text = str(root).replace(os.sep, "/")
        self.source = source
        self.source_lines = source.splitlines()
        self._segment_cache = {}
        self.rel = path.relative_to(root).as_posix()
        self.tree = ast.parse(source, filename=str(path))
        self.parents = {}
        for node in ast.walk(self.tree):
            for child in ast.iter_child_nodes(node):
                self.parents[id(child)] = node
        self.self_path = str(path.resolve()).replace(os.sep, "/")
        self.arg_specs = []
        self.arg_paths = {}
        self.arg_explicit_none = set()
        self.path_env = {}
        self.listing_env = {}
        self.elem_env = {}
        self.func_listing = {}
        self.imported_names = {}
        self.arg_namespaces = set(ARG_NAMESPACES)
        self._scan_namespaces()
        self._scan_functions()
        self._scan_imports()
        self._collect_arguments()
        for _ in range(3):
            self._collect_bindings()
            self._collect_arg_paths()

    def segment(self, node):
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        col = getattr(node, "col_offset", None)
        end_col = getattr(node, "end_col_offset", None)
        if start is None or end is None or col is None or end_col is None:
            return ""
        key = (start, col, end, end_col)
        cached = self._segment_cache.get(key)
        if cached is not None:
            return cached
        if start > len(self.source_lines) or end > len(self.source_lines):
            return ""
        if start == end:
            text = self.source_lines[start - 1][col:end_col]
        else:
            chunks = [self.source_lines[start - 1][col:]]
            chunks.extend(self.source_lines[start:end - 1])
            chunks.append(self.source_lines[end - 1][:end_col])
            text = "\n".join(chunks)
        self._segment_cache[key] = text
        return text

    def _scan_namespaces(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            if not (isinstance(func, ast.Attribute) and func.attr in ("parse_args", "parse_known_args")):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.arg_namespaces.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for element in target.elts:
                        if isinstance(element, ast.Name):
                            self.arg_namespaces.add(element.id)

    def _scan_imports(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.imported_names[local] = (node.module, alias.name, node.lineno)

    def _scan_functions(self):
        for node in self.tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = [arg.arg for arg in node.args.args]
            if not params:
                continue
            for inner in ast.walk(node):
                found = self._raw_listing(inner, params)
                if found:
                    self.func_listing[node.name] = found
                    break

    def _raw_listing(self, node, params):
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in LISTING_METHODS:
            return None
        if not isinstance(func.value, ast.Name) or func.value.id not in params:
            return None
        pattern = None
        if node.args and isinstance(node.args[0], ast.Constant):
            value = node.args[0].value
            pattern = value if isinstance(value, str) else None
        return params.index(func.value.id), func.attr, pattern

    def _collect_arguments(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
                continue
            names = [
                arg.value for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if not names:
                continue
            spec = {
                "names": names,
                "positional": not names[0].startswith("-"),
                "line": node.lineno,
                "choices": None,
                "default": None,
                "default_expr": None,
                "type": None,
                "required": None,
                "action": None,
                "nargs": None,
                "help": None,
            }
            dest = None
            default_node = None
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                if kw.arg == "choices":
                    spec["choices"] = self._literal_sequence(kw.value)
                elif kw.arg == "default":
                    default_node = kw.value
                    spec["default"] = self._literal_scalar(kw.value)
                    spec["default_expr"] = self.segment(kw.value)
                elif kw.arg == "type":
                    spec["type"] = self.segment(kw.value)
                elif kw.arg in ("required", "action", "nargs", "help"):
                    spec[kw.arg] = self._literal_scalar(kw.value)
                elif kw.arg == "dest":
                    dest = self._literal_scalar(kw.value)
            if dest is None:
                head = names[0]
                dest = head.lstrip("-").replace("-", "_") if head.startswith("-") else head
            spec["dest"] = dest
            spec["_default_node"] = default_node
            if isinstance(default_node, ast.Constant) and default_node.value is None:
                self.arg_explicit_none.add(dest)
            self.arg_specs.append(spec)

    def _collect_arg_paths(self):
        for spec in self.arg_specs:
            dest = spec["dest"]
            node = spec.get("_default_node")
            resolved = self.resolve(node) if node is not None else None
            if resolved is None and dest not in self.arg_explicit_none:
                pathy = "Path" in (spec.get("type") or "")
                if pathy or spec["positional"] or spec.get("required") is True:
                    resolved = "<%s>" % dest.upper()
            if resolved is not None:
                self.arg_paths[dest] = resolved
            spec["resolved_default"] = None if is_templated(resolved) else resolved

    def _collect_bindings(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
                if len(targets) == 1:
                    name = targets[0].id
                    resolved = self.resolve(node.value)
                    if resolved:
                        self.path_env[name] = resolved
                    listing = self.listing_source(node.value)
                    if listing:
                        self.listing_env[name] = listing
            elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
                listing = self.listing_source(node.iter)
                if listing:
                    base = self.resolve(listing[0])
                    if base:
                        joined = join_path(base, listing[2] or "*")
                        if joined:
                            self.elem_env[node.target.id] = joined

    def resolve(self, node, depth=0):
        if node is None or depth > 14:
            return None
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.Name):
            if node.id == "__file__":
                return self.self_path
            if node.id in self.path_env:
                return self.path_env[node.id]
            return self.elem_env.get(node.id)
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in self.arg_namespaces:
                return self.arg_paths.get(node.attr)
            if node.attr == "parent":
                return parent_path(self.resolve(node.value, depth + 1))
            if node.attr in ("name", "stem"):
                base = self.resolve(node.value, depth + 1)
                if base:
                    tail = PurePosixPath(base).name
                    return tail if node.attr == "name" else PurePosixPath(tail).stem
            return None
        if isinstance(node, ast.Subscript):
            holder = node.value
            index = node.slice
            if isinstance(index, ast.Index):
                index = index.value
            if (
                isinstance(holder, ast.Attribute)
                and holder.attr == "parents"
                and isinstance(index, ast.Constant)
                and isinstance(index.value, int)
            ):
                base = self.resolve(holder.value, depth + 1)
                if base:
                    ancestors = list(PurePosixPath(base).parents)
                    if index.value < len(ancestors):
                        return str(ancestors[index.value])
            return None
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr in ("resolve", "absolute", "expanduser", "as_posix"):
                    return self.resolve(func.value, depth + 1)
                if func.attr == "joinpath":
                    base = self.resolve(func.value, depth + 1)
                    for arg in node.args:
                        base = join_path(base, self.resolve(arg, depth + 1))
                    return base
            if isinstance(func, ast.Name) and func.id in PATH_FACTORIES + ("str",):
                if node.args:
                    base = self.resolve(node.args[0], depth + 1)
                    for arg in node.args[1:]:
                        base = join_path(base, self.resolve(arg, depth + 1))
                    return base
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return join_path(
                self.resolve(node.left, depth + 1), self.resolve(node.right, depth + 1)
            )
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            candidates = [self.resolve(value, depth + 1) for value in node.values]
            concrete = [item for item in candidates if item and not is_templated(item)]
            if concrete:
                return concrete[0]
            for item in candidates:
                if item:
                    return item
            return None
        if isinstance(node, ast.IfExp):
            return self.resolve(node.body, depth + 1) or self.resolve(node.orelse, depth + 1)
        if isinstance(node, ast.JoinedStr):
            chunks = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    chunks.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    inner = self.resolve(value.value, depth + 1)
                    chunks.append(inner if inner else "{" + self.segment(value.value) + "}")
            joined = "".join(chunks)
            return norm_path(joined) if joined else None
        return None

    def listing_source(self, node, depth=0):
        if node is None or depth > 10:
            return None
        if isinstance(node, ast.Name):
            return self.listing_env.get(node.id)
        if isinstance(node, (ast.SetComp, ast.ListComp, ast.GeneratorExp, ast.DictComp)):
            for generator in node.generators:
                found = self.listing_source(generator.iter, depth + 1)
                if found:
                    return found
            return None
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            for elt in node.elts:
                found = self.listing_source(elt, depth + 1)
                if found:
                    return found
            return None
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in WRAPPERS and node.args:
                return self.listing_source(node.args[0], depth + 1)
            if isinstance(func, ast.Attribute):
                if func.attr in LISTING_METHODS:
                    pattern = None
                    if node.args and isinstance(node.args[0], ast.Constant):
                        value = node.args[0].value
                        pattern = value if isinstance(value, str) else None
                    return (func.value, func.attr, pattern, node.lineno)
                if func.attr in ("walk", "listdir", "scandir") and node.args:
                    return (node.args[0], func.attr, None, node.lineno)
            if isinstance(func, ast.Name) and func.id in self.func_listing and node.args:
                index, method, pattern = self.func_listing[func.id]
                if index < len(node.args):
                    return (node.args[index], method, pattern, node.lineno)
            return None
        return None

    def _literal_scalar(self, node):
        try:
            return ast.literal_eval(node)
        except Exception:
            return self.segment(node) or None

    def _literal_sequence(self, node):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            out = []
            for elt in node.elts:
                try:
                    out.append(ast.literal_eval(elt))
                except Exception:
                    out.append(self.segment(elt))
            return out
        try:
            value = ast.literal_eval(node)
        except Exception:
            return None
        if isinstance(value, (list, tuple, set, frozenset)):
            return sorted(value, key=repr)
        return None

    def contains(self, haystack, needle):
        for node in ast.walk(haystack):
            if node is needle:
                return True
        return False

    def scope_label(self, absolute):
        if not absolute:
            return None
        if absolute == self.root_text:
            return "repo_root"
        if absolute.startswith(self.root_text + "/"):
            return absolute[len(self.root_text) + 1:]
        return absolute


def constant_token(name):
    match = CONST_PREFIX_RE.match(name)
    if match:
        return match.group(1)
    match = CONST_SUFFIX_RE.match(name)
    if match:
        return match.group(1)
    return None


def sequence_literal(node):
    if isinstance(node, ast.Set):
        return "set", list(node.elts)
    if isinstance(node, ast.List):
        return "list", list(node.elts)
    if isinstance(node, ast.Tuple):
        return "tuple", list(node.elts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in ("frozenset", "set", "list", "tuple") and len(node.args) == 1:
            inner = sequence_literal(node.args[0])
            if inner:
                return node.func.id, inner[1]
    return None


def element_value(node, namespaces=ARG_NAMESPACES):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, "str"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in PATH_FACTORIES:
            parts = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    parts.append(arg.value)
                else:
                    return None, "expr"
            if parts:
                return "/".join(parts), "Path"
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id in namespaces:
            return None, "arg:%s" % node.attr
    return None, "expr"


def looks_path_like(values, forms, scope=None):
    if not values:
        return False
    if "Path" in forms:
        return True
    if scope and scope != "unknown":
        return True
    hits = sum(
        1 for value in values
        if "/" in value or has_file_ext(value) or value.startswith(".")
    )
    return hits * 2 >= len(values)


def extension_counts(values):
    counts = {}
    for value in values:
        suffix = PurePosixPath(value).suffix
        if suffix:
            counts[suffix] = counts.get(suffix, 0) + 1
    return dict(sorted(counts.items()))


def constant_effect(token, scope, exact, relaxed):
    where = "仓库根" if scope == "repo_root" else (
        scope if scope != "unknown" else "未能推断的作用域"
    )
    if token in ("ALLOWED", "ALLOWLIST", "WHITELIST", "PERMITTED"):
        text = f"{where}下出现不在此集合的条目会导致该门禁失败"
    elif token == "REQUIRED":
        text = f"{where}下必须存在此集合中的全部条目，缺失会导致该门禁失败"
    elif token == "OPTIONAL":
        text = f"{where}下这些条目允许不存在；存在时必须是常规非符号链接目录"
    elif token == "EXPECTED":
        text = f"{where}下的实际内容必须与此集合一致，否则该门禁失败"
    elif token in ("FORBIDDEN", "DENIED", "BLACKLIST", "DENYLIST"):
        text = f"{where}下出现此集合中的任何条目会导致该门禁失败"
    elif token == "IMMUTABLE":
        text = f"{where}下这些条目被声明为不可修改，改动会导致该门禁失败"
    else:
        text = f"{where}下的内容受此集合约束"
    if exact:
        text += "；该常量同时参与双向差集或相等比较，等价于要求集合精确相等"
    if relaxed:
        text += "；豁免项来自 " + ", ".join(relaxed)
    return text


def operand_nodes(node):
    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    if isinstance(node, ast.Compare):
        return [node.left, *node.comparators]
    if isinstance(node, ast.BoolOp):
        return list(node.values)
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        return list(node.elts)
    if isinstance(node, ast.Call):
        out = list(node.args) + [kw.value for kw in node.keywords]
        if isinstance(node.func, ast.Attribute):
            out.append(node.func.value)
        return out
    return []


def analyze_constant_usage(module, name):
    usage = {
        "lines": [],
        "scopes": [],
        "left_of_sub": False,
        "right_of_sub": False,
        "equality": False,
        "relaxed": [],
    }
    for node in ast.walk(module.tree):
        if not (isinstance(node, ast.Name) and node.id == name):
            continue
        if not isinstance(node.ctx, ast.Load):
            continue
        parent_of_name = module.parents.get(id(node))
        if isinstance(parent_of_name, (ast.Import, ast.ImportFrom)):
            continue
        usage["lines"].append(node.lineno)
        current = node
        for _ in range(6):
            parent = module.parents.get(id(current))
            if parent is None:
                break
            if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Sub):
                if module.contains(parent.left, node):
                    usage["left_of_sub"] = True
                else:
                    usage["right_of_sub"] = True
                for sibling in (parent.left, parent.right):
                    if module.contains(sibling, node):
                        continue
                    if isinstance(sibling, ast.Name) and sibling.id.isupper():
                        if sibling.id != name and sibling.id not in usage["relaxed"]:
                            usage["relaxed"].append(sibling.id)
            if isinstance(parent, ast.Compare):
                if any(isinstance(op, (ast.Eq, ast.NotEq)) for op in parent.ops):
                    usage["equality"] = True
            if isinstance(
                parent,
                (ast.BinOp, ast.Compare, ast.Call, ast.BoolOp, ast.Set, ast.Tuple, ast.List),
            ):
                for sibling in operand_nodes(parent):
                    if module.contains(sibling, node):
                        continue
                    listing = module.listing_source(sibling)
                    if not listing:
                        continue
                    label = module.scope_label(module.resolve(listing[0]))
                    if label:
                        usage["scopes"].append((node.lineno, label))
            current = parent
    for node in ast.walk(module.tree):
        if not isinstance(node, (ast.For, ast.AsyncFor)):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if not any(
            isinstance(inner, ast.Name) and inner.id == name for inner in ast.walk(node.iter)
        ):
            continue
        loop_var = node.target.id
        for stmt in node.body:
            for inner in ast.walk(stmt):
                if (
                    isinstance(inner, ast.BinOp)
                    and isinstance(inner.op, ast.Div)
                    and isinstance(inner.right, ast.Name)
                    and inner.right.id == loop_var
                ):
                    label = module.scope_label(module.resolve(inner.left))
                    if label:
                        usage["scopes"].append((inner.lineno, label))
    return usage


def pick_scope(usage):
    candidates = [{"line": line, "scope": scope} for line, scope in usage["scopes"]]
    if not candidates:
        return "unknown", candidates
    tally = {}
    order = {}
    for index, (_, scope) in enumerate(usage["scopes"]):
        tally[scope] = tally.get(scope, 0) + 1
        order.setdefault(scope, index)
    best = sorted(tally.items(), key=lambda item: (-item[1], order[item[0]]))[0][0]
    return best, candidates


def extract_path_constraints(modules):
    definitions = {}
    records = []
    for rel in sorted(modules):
        module = modules[rel]
        for node in module.tree.body:
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
                value = node.value
            else:
                continue
            if len(targets) != 1 or value is None:
                continue
            name = targets[0].id
            token = constant_token(name)
            if not token:
                continue
            literal = sequence_literal(value)
            if not literal:
                continue
            container, elements = literal
            values = []
            forms = []
            unresolved = []
            for element in elements:
                text, form = element_value(element, module.arg_namespaces)
                if form not in forms:
                    forms.append(form)
                if text is None:
                    unresolved.append(module.segment(element) or form)
                else:
                    values.append(text)
            if not values:
                continue
            usage = analyze_constant_usage(module, name)
            scope, scope_candidates = pick_scope(usage)
            exact = (usage["left_of_sub"] and usage["right_of_sub"]) or usage["equality"]
            record = {
                "gate": str(module.path),
                "constant": name,
                "kind": KIND_BY_TOKEN.get(token, "unknown"),
                "scope": scope,
                "values": values,
                "line": node.lineno,
                "effect": constant_effect(token, scope, exact, usage["relaxed"]),
                "name_token": token,
                "container": container,
                "element_forms": forms,
                "value_count": len(values),
                "value_extension_counts": extension_counts(values),
                "unresolved_elements": unresolved,
                "looks_path_like": looks_path_like(values, forms, scope),
                "exact_set": bool(exact),
                "enforces_presence": bool(usage["left_of_sub"]),
                "enforces_absence": bool(usage["right_of_sub"]),
                "relaxed_by": usage["relaxed"],
                "usage_lines": sorted(set(usage["lines"])),
                "scope_candidates": scope_candidates,
                "definition": "literal",
                "defined_in": None,
                "also_enforced_by": [],
            }
            definitions[(Path(rel).stem, name)] = record
            records.append(record)
    for rel in sorted(modules):
        module = modules[rel]
        for local, (source_module, original, lineno) in module.imported_names.items():
            token = constant_token(original)
            if not token:
                continue
            key = (source_module.split(".")[-1], original)
            base = definitions.get(key)
            if base is None:
                continue
            usage = analyze_constant_usage(module, local)
            if not usage["lines"]:
                continue
            scope, scope_candidates = pick_scope(usage)
            exact = (usage["left_of_sub"] and usage["right_of_sub"]) or usage["equality"]
            base["also_enforced_by"].append(str(module.path))
            records.append({
                "gate": str(module.path),
                "constant": local,
                "kind": base["kind"],
                "scope": scope,
                "values": list(base["values"]),
                "line": lineno,
                "effect": constant_effect(token, scope, exact, usage["relaxed"]),
                "name_token": token,
                "container": base["container"],
                "element_forms": list(base["element_forms"]),
                "value_count": base["value_count"],
                "value_extension_counts": dict(base["value_extension_counts"]),
                "unresolved_elements": [],
                "looks_path_like": base["looks_path_like"],
                "exact_set": bool(exact),
                "enforces_presence": bool(usage["left_of_sub"]),
                "enforces_absence": bool(usage["right_of_sub"]),
                "relaxed_by": usage["relaxed"],
                "usage_lines": sorted(set(usage["lines"])),
                "scope_candidates": scope_candidates,
                "definition": "imported",
                "defined_in": base["gate"],
                "also_enforced_by": [],
            })
    return records


def collect_path_expressions(module):
    found = {}
    for node in ast.walk(module.tree):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)):
            continue
        resolved = module.resolve(node)
        if not resolved or not resolved.startswith("/"):
            continue
        if is_templated(resolved):
            continue
        found.setdefault(resolved, node.lineno)
    return found


def common_tail(a, b):
    left = [part for part in a.split("/") if part]
    right = [part for part in b.split("/") if part]
    tail = []
    while left and right and left[-1] == right[-1]:
        tail.insert(0, left.pop())
        right.pop()
    return tail


def comparison_signals(module):
    signals = []
    source = module.source
    dump_tool = DUMP_TOOL_RE.search(source)
    for node in ast.walk(module.tree):
        if not isinstance(node, ast.Compare):
            continue
        if not any(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            continue
        sides = [module.segment(node.left)] + [module.segment(c) for c in node.comparators]
        joined = " ".join(sides)
        if len(sides) < 2 or not all(sides):
            continue
        if "sass" in joined.lower() and dump_tool:
            if "sass_byte_identical" not in signals:
                signals.append("sass_byte_identical")
        if "resource" in joined.lower():
            if "ptxas_resource_counts" not in signals:
                signals.append("ptxas_resource_counts")
        if "cubin" in joined.lower():
            if "cubin_byte_identical" not in signals:
                signals.append("cubin_byte_identical")
        if re.search(r"sha256|digest|hexdigest", joined, re.IGNORECASE):
            if "digest_equal" not in signals:
                signals.append("digest_equal")
    if re.search(r"sha256\s*\(\s*[\w.]*cubin", source, re.IGNORECASE):
        if "cubin_sha256" not in signals:
            signals.append("cubin_sha256")
    if "ptxas" in source and re.search(r"registers|spill|barriers|stack frame", source):
        if "ptxas_resource_counts" not in signals:
            signals.append("ptxas_resource_counts")
    return signals, bool(dump_tool), (dump_tool.group(0) if dump_tool else None)


def extract_mirrored_trees(modules, root):
    root_text = str(root).replace(os.sep, "/")
    results = []
    for rel in sorted(modules):
        module = modules[rel]
        signals, has_dump, dump_tool = comparison_signals(module)
        has_hash = bool(re.search(r"sha256|hashlib|filecmp|cmp\b", module.source))
        binary_tokens = bool(BINARY_TOKEN_RE.search(module.source))
        if not ((has_dump or (has_hash and binary_tokens)) and signals):
            continue
        expressions = collect_path_expressions(module)
        trees = {
            path: line for path, line in expressions.items()
            if not has_file_ext(path)
        }
        include_flag = bool(re.search(r"-I\{|-I\"|-I'|\"-I|'-I|-isystem", module.source))
        pairs = []
        ordered = sorted(trees)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if a == b or a.startswith(b + "/") or b.startswith(a + "/"):
                    continue
                tail = common_tail(a, b)
                if len(tail) < 2:
                    continue
                hint = any(part in TREE_HINT_COMPONENTS for part in tail)
                if not (hint or include_flag):
                    continue
                pairs.append((a, b, tail))
        for a, b, tail in pairs:
            a_outside = not (a == root_text or a.startswith(root_text + "/"))
            b_outside = not (b == root_text or b.startswith(root_text + "/"))
            if b_outside and not a_outside:
                tree_a, tree_b = a, b
            elif a_outside and not b_outside:
                tree_a, tree_b = b, a
            else:
                tree_a, tree_b = a, b
            a_outside = not (tree_a == root_text or tree_a.startswith(root_text + "/"))
            b_outside = not (tree_b == root_text or tree_b.startswith(root_text + "/"))
            primary = "sass_byte_identical" if "sass_byte_identical" in signals else (
                "cubin_byte_identical" if "cubin_byte_identical" in signals else (
                    "cubin_sha256" if "cubin_sha256" in signals else "binary_identical"
                )
            )
            effect = "改动任一侧的源文件必须在另一侧做等价改动，否则门禁失败"
            if "ptxas_resource_counts" in signals:
                effect += "；比对同时覆盖 ptxas 资源计数（寄存器/barrier/栈帧/spill）"
            if b_outside:
                effect += f"；镜像的另一侧位于 --root 之外（{tree_b}），改它属于修改工作目录之外的内容"
            results.append({
                "gate": str(module.path),
                "tree_a": tree_a,
                "tree_b": tree_b,
                "comparison": primary,
                "line": min(trees[tree_a], trees[tree_b]),
                "effect": effect,
                "tree_b_outside_repo": b_outside,
                "tree_a_outside_repo": a_outside,
                "tree_a_line": trees[tree_a],
                "tree_b_line": trees[tree_b],
                "tree_a_relative_to_root": os.path.relpath(tree_a, root_text),
                "tree_b_relative_to_root": os.path.relpath(tree_b, root_text),
                "shared_suffix": "/".join(tail),
                "comparison_signals": signals,
                "dump_tool": dump_tool,
            })
    return results


def target_requirements(module, dest):
    requirements = []
    prefix = tuple(f"{name}.{dest}" for name in sorted(module.arg_namespaces))
    for node in ast.walk(module.tree):
        if isinstance(node, ast.Compare):
            left_text = module.segment(node.left)
            if left_text.startswith(prefix) and left_text.endswith(".suffix"):
                for op, comparator in zip(node.ops, node.comparators):
                    literal = module._literal_scalar(comparator)
                    if isinstance(literal, str):
                        if isinstance(op, ast.NotEq):
                            requirements.append(f"后缀必须是 {literal}")
                        elif isinstance(op, ast.Eq):
                            requirements.append(f"后缀不得是 {literal}")
            if any(isinstance(op, ast.In) for op in node.ops):
                container = module.segment(node.comparators[0]) if node.comparators else ""
                literal = module._literal_scalar(node.left)
                if container.startswith(prefix) and literal == "..":
                    requirements.append('路径不得包含 ".." 分量')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = module.segment(node.func.value)
            if owner.startswith(prefix) and node.func.attr == "is_absolute":
                requirements.append("必须是相对路径")
    seen = []
    for item in requirements:
        if item not in seen:
            seen.append(item)
    return seen


def extract_diff_constraints(modules, root):
    results = []
    for rel in sorted(modules):
        module = modules[rel]
        change_vars = set()
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and DIFF_CHANGE_RE.match(target.id):
                        change_vars.add(target.id)
        for node in ast.walk(module.tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            if len(targets) != 1:
                continue
            name = targets[0].id
            literal = sequence_literal(node.value)
            if not literal:
                continue
            container, elements = literal
            name_hit = bool(DIFF_ALLOW_RE.match(name))
            usage_hit = False
            for other in ast.walk(module.tree):
                if not (isinstance(other, ast.BinOp) and isinstance(other.op, ast.Sub)):
                    continue
                right = other.right
                left = other.left
                if isinstance(right, ast.Name) and right.id == name:
                    if isinstance(left, ast.Name) and left.id in change_vars:
                        usage_hit = True
            if not (name_hit or usage_hit):
                continue
            patterns = []
            arg_dests = []
            for element in elements:
                text, form = element_value(element, module.arg_namespaces)
                if text is not None:
                    patterns.append(text)
                elif form.startswith("arg:"):
                    dest = form.split(":", 1)[1]
                    arg_dests.append(dest)
                    flag = None
                    for spec in module.arg_specs:
                        if spec["dest"] == dest:
                            flag = spec["names"][0]
                            break
                    patterns.append(f"<{flag}>" if flag else f"<{dest.upper()}>")
                else:
                    patterns.append(module.segment(element) or "<expr>")
            must_change = []
            for other in ast.walk(module.tree):
                if not isinstance(other, ast.Compare):
                    continue
                if not any(isinstance(op, ast.NotIn) for op in other.ops):
                    continue
                container_text = module.segment(other.comparators[0]) if other.comparators else ""
                if container_text not in change_vars:
                    continue
                text, form = element_value(other.left, module.arg_namespaces)
                if text is not None:
                    must_change.append(text)
                elif form.startswith("arg:"):
                    dest = form.split(":", 1)[1]
                    flag = None
                    for spec in module.arg_specs:
                        if spec["dest"] == dest:
                            flag = spec["names"][0]
                            break
                    must_change.append(f"<{flag}>" if flag else f"<{dest.upper()}>")
            scopes = []
            for candidate in ("base", "candidate", "left", "right", "before", "after"):
                resolved = module.path_env.get(candidate)
                label = module.scope_label(resolved) if resolved else None
                if label:
                    scopes.append({"name": candidate, "scope": label})
            requirements = {}
            for dest in arg_dests:
                items = target_requirements(module, dest)
                if items:
                    requirements[dest] = items
            count_limits = []
            for other in ast.walk(module.tree):
                if not isinstance(other, ast.Compare):
                    continue
                left_text = module.segment(other.left)
                if not re.match(r"^len\(", left_text):
                    continue
                inner = left_text[4:-1]
                if inner not in change_vars:
                    continue
                for op, comparator in zip(other.ops, other.comparators):
                    limit = module._literal_scalar(comparator)
                    if isinstance(limit, int):
                        count_limits.append({
                            "expr": left_text,
                            "op": type(op).__name__,
                            "limit": limit,
                            "line": other.lineno,
                        })
            effect = (
                f"两棵版本树之间只允许 {len(patterns)} 个文件发生变更："
                + ", ".join(patterns)
                + "；出现其它变更文件该门禁直接失败"
            )
            if must_change:
                effect += "；且 " + ", ".join(must_change) + " 必须发生变更"
            results.append({
                "gate": str(module.path),
                "variable": name,
                "container": container,
                "line": node.lineno,
                "allowed_patterns": patterns,
                "max_changed_files": len(patterns),
                "must_change": must_change,
                "argument_dests": arg_dests,
                "argument_requirements": requirements,
                "compared_scopes": scopes,
                "count_limits": count_limits,
                "effect": effect,
            })
    return results


GENERIC_TEXT_RE = re.compile("|".join(GENERIC_TEXT_KEYWORDS), re.IGNORECASE)
CATEGORY_COMPILED = tuple(
    (name, re.compile("|".join(patterns), re.IGNORECASE))
    for name, patterns in TEXT_CATEGORY_RULES
)


def split_blocks(lines):
    blocks = []
    current = []
    for index, raw in enumerate(lines, start=1):
        if not raw.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        if current and BLOCK_BREAK_RE.match(raw):
            blocks.append(current)
            current = []
        current.append((index, raw))
    if current:
        blocks.append(current)
    return blocks


def block_sentences(block):
    pieces = []
    offset = 0
    spans = {}
    for lineno, raw in block:
        text = raw.strip()
        spans[lineno] = (offset, offset + len(text))
        pieces.append(text)
        offset += len(text) + 1
    joined = " ".join(pieces)
    sentences = []
    cursor = 0
    for chunk in SENTENCE_SPLIT_RE.split(joined):
        start = joined.find(chunk, cursor)
        if start < 0:
            start = cursor
        sentences.append((start, start + len(chunk), chunk))
        cursor = start + len(chunk)
    return joined, spans, sentences


def path_subjects(text):
    subjects = []
    for token in BACKTICK_RE.findall(text):
        candidate = token.strip()
        if not candidate:
            continue
        if " " in candidate:
            continue
        if "/" in candidate or candidate.startswith(".") or has_file_ext(candidate):
            if candidate not in subjects:
                subjects.append(candidate)
    for token in BARE_TMP_RE.findall(text):
        if token not in subjects:
            subjects.append(token)
    return subjects


def gate_mentions(text, gate_stems):
    hits = []
    for stem in gate_stems:
        if stem and stem in text and stem not in hits:
            hits.append(stem)
    return hits


def categorize(text):
    matched = []
    for name, pattern in CATEGORY_COMPILED:
        if pattern.search(text):
            matched.append(name)
    return (matched[0] if matched else "other"), matched


def extract_textual_constraints(files, root, gate_stems):
    records = []
    contracts = [
        path for path in files
        if path.name in CONTRACT_FILENAMES
    ]
    contracts.sort(key=lambda path: (len(path.relative_to(root).parts), str(path)))
    for path in contracts:
        source = read_text(path)
        if source is None:
            continue
        rel = path.relative_to(root).as_posix()
        is_root_contract = "/" not in rel
        lines = source.splitlines()
        for block in split_blocks(lines):
            joined, spans, sentences = block_sentences(block)
            for lineno, raw in block:
                text = raw.strip()
                if not text:
                    continue
                offsets = []
                generic = GENERIC_TEXT_RE.search(text)
                if generic:
                    offsets.append(generic.start())
                for _, pattern in CATEGORY_COMPILED:
                    hit = pattern.search(text)
                    if hit:
                        offsets.append(hit.start())
                if not offsets:
                    continue
                start, end = spans[lineno]
                anchor = start + min(offsets)
                sentence = joined
                for s_start, s_end, chunk in sentences:
                    if s_start <= anchor < s_end:
                        sentence = chunk
                        break
                else:
                    for s_start, s_end, chunk in sentences:
                        if s_start < end and s_end > start:
                            sentence = chunk
                            break
                category, categories = categorize(text)
                sentence_category, sentence_categories = categorize(sentence)
                if category == "other" and sentence_category != "other":
                    category = sentence_category
                    categories = sentence_categories
                records.append({
                    "file": str(path),
                    "file_relative": rel,
                    "line": lineno,
                    "text": text,
                    "category": category,
                    "categories": categories,
                    "sentence": sentence,
                    "subjects": path_subjects(text),
                    "sentence_subjects": path_subjects(sentence),
                    "gate_mentions": gate_mentions(sentence, gate_stems),
                    "source": f"{rel}:{lineno}",
                    "is_root_contract": is_root_contract,
                })
    return records


VERSION_GLOB_RE = re.compile(r"v\[0-9\]\*|v\[0-9\]|v\*|vN|v\{[^}]*\}")
VERSION_PLACEHOLDER_RE = re.compile(r"<(?:BASE|CANDIDATE|VERSION|VN|V)>")


def canonical_zone_path(text, root):
    value = text.strip()
    if value.startswith(str(root)):
        value = os.path.relpath(value, str(root))
    value = VERSION_PLACEHOLDER_RE.sub("vN", value)
    value = VERSION_GLOB_RE.sub("vN", value)
    value = value.rstrip("/")
    return value or "."


def as_directory(text):
    return text if text.endswith("/") else text + "/"


def zone_path_exists(root, rel):
    if not rel:
        return False
    try:
        if (root / rel).exists():
            return True
    except OSError:
        return False
    if "vN" in rel:
        try:
            return any(True for _ in root.glob(rel.replace("vN", "v*")))
        except (OSError, ValueError):
            return False
    return False


def make_subject_resolver(root, known_names):
    def resolve(subject, contract_relative):
        raw = (subject or "").strip()
        if not raw or raw.startswith("/") or raw.startswith("~"):
            return None
        if raw.startswith("./"):
            raw = raw[2:]
        trailing = raw.endswith("/")
        core = canonical_zone_path(raw.strip("/"), root)
        if not core or core == ".":
            return None
        contract_dir = str(PurePosixPath(contract_relative).parent)
        contract_dir = "" if contract_dir == "." else contract_dir
        candidates = []
        if contract_dir:
            candidates.append(f"{contract_dir}/{core}")
        candidates.append(core)
        for candidate in candidates:
            if zone_path_exists(root, candidate):
                return as_directory(candidate) if (
                    trailing or (root / candidate).is_dir()
                ) else candidate
        head = core.split("/")[0]
        if head in known_names:
            return as_directory(core) if trailing else core
        return None

    return resolve


def build_write_zones(root, path_constraints, mirrored, diff_constraints, textual):
    root_entries = set()
    try:
        root_entries = {child.name for child in root.iterdir()}
    except OSError:
        pass

    allow_root, optional_root, required_root, forbidden_root = set(), set(), set(), set()
    dir_valued = set()
    constraint_by_value = {}
    for record in path_constraints:
        if record["scope"] != "repo_root" or not record["looks_path_like"]:
            continue
        token = record["name_token"]
        bucket = None
        if token in ("ALLOWED", "ALLOWLIST", "WHITELIST", "PERMITTED", "EXPECTED"):
            bucket = allow_root
        elif token == "REQUIRED":
            bucket = required_root
            allow_root.update(record["values"])
        elif token == "OPTIONAL":
            bucket = optional_root
            allow_root.update(record["values"])
        elif token in ("FORBIDDEN", "DENIED", "BLACKLIST", "DENYLIST"):
            bucket = forbidden_root
        if bucket is None:
            continue
        bucket.update(record["values"])
        if re.search(r"DIRECTOR|DIRS?$|_DIRS", record["constant"]):
            dir_valued.update(record["values"])
        for value in record["values"]:
            constraint_by_value.setdefault(value, []).append(record)

    known_names = set(root_entries) | allow_root | optional_root | required_root | forbidden_root
    resolve_subject = make_subject_resolver(root, known_names)
    self_reference = re.compile(
        r"this (?:directory|dir|folder|tree|file)|本目录|该目录|本文件|该文件", re.IGNORECASE
    )

    def record_targets(record):
        subjects = record["sentence_subjects"] or record["subjects"]
        targets = []
        for subject in subjects:
            resolved = resolve_subject(subject, record["file_relative"])
            if resolved and resolved not in targets:
                targets.append(resolved)
        if not targets and not record["is_root_contract"]:
            if self_reference.search(record["sentence"]):
                owner = str(PurePosixPath(record["file_relative"]).parent)
                if owner and owner != ".":
                    targets.append(as_directory(owner))
        return targets

    immutable = []
    immutable_keys = set()
    for record in textual:
        if record["category"] != "immutable_path":
            continue
        for subject in record_targets(record):
            key = canonical_zone_path(subject, root)
            if key in immutable_keys:
                for entry in immutable:
                    if entry["path"].rstrip("/") != key:
                        continue
                    if record["source"] not in entry["also_stated_in"]:
                        entry["also_stated_in"].append(record["source"])
                continue
            immutable_keys.add(key)
            immutable.append({
                "path": subject,
                "source": record["source"],
                "text": record["text"],
                "sentence": record["sentence"],
                "also_stated_in": [],
            })

    gated = []

    def directory_display(key, hint_dir):
        if hint_dir or key in dir_valued:
            return as_directory(key)
        probe = root / key
        if probe.is_dir():
            return as_directory(key)
        if "vN" in key:
            try:
                for candidate in root.glob(key.replace("vN", "v*")):
                    if candidate.is_dir():
                        return as_directory(key)
            except (OSError, ValueError):
                pass
        return key

    def add_gated(path_value, gate, requirement, source=None, text=None):
        key = canonical_zone_path(path_value, root)
        if key in immutable_keys:
            return
        display = directory_display(key, path_value.endswith("/"))
        for entry in gated:
            if entry["_key"] != key:
                continue
            if display.endswith("/") and not entry["path"].endswith("/"):
                entry["path"] = display
            if gate and gate not in entry["gates"]:
                entry["gates"].append(gate)
            if requirement not in entry["requirements"]:
                entry["requirements"].append(requirement)
            if source and source not in entry["sources"]:
                entry["sources"].append(source)
            entry["requirement"] = "；".join(entry["requirements"])
            if entry["gate"] is None and gate:
                entry["gate"] = gate
            if entry["text"] is None and text:
                entry["text"] = text
            return
        gated.append({
            "_key": key,
            "path": display,
            "gate": gate,
            "requirement": requirement,
            "gates": [gate] if gate else [],
            "requirements": [requirement],
            "sources": [source] if source else [],
            "text": text,
        })

    presence_enforced = any(
        record["enforces_presence"]
        for record in path_constraints
        if record["scope"] == "repo_root"
        and record["looks_path_like"]
        and record["name_token"] in ("ALLOWED", "ALLOWLIST", "WHITELIST", "PERMITTED", "EXPECTED")
    )
    mandatory_root = sorted(allow_root - optional_root) if presence_enforced else []
    allow_gate = None
    for record in path_constraints:
        if record["scope"] == "repo_root" and record["name_token"] in ("ALLOWED", "ALLOWLIST"):
            allow_gate = record["gate"]
            break
    for name in mandatory_root:
        add_gated(
            name,
            allow_gate,
            "structure_gate 要求该根条目必须存在（ALLOWED_ROOT_NAMES 减去 OPTIONAL_ROOT_DIRECTORIES）",
        )
    for name in sorted(required_root):
        gate = None
        for record in constraint_by_value.get(name, []):
            if record["name_token"] == "REQUIRED":
                gate = record["gate"]
        add_gated(name, gate, "必须是常规非符号链接文件")
    for name in sorted(forbidden_root):
        gate = None
        for record in constraint_by_value.get(name, []):
            if record["name_token"] in ("FORBIDDEN", "DENIED", "BLACKLIST", "DENYLIST"):
                gate = record["gate"]
        add_gated(name, gate, "禁止在仓库根创建该条目（FORBIDDEN 名单）")

    for record in path_constraints:
        if record["scope"] in ("repo_root", "unknown") or not record["looks_path_like"]:
            continue
        if record["scope"].startswith("/"):
            continue
        extensions = ", ".join(
            f"{count} 个 {ext}" for ext, count in record["value_extension_counts"].items()
        )
        requirement = (
            f"目录内容必须精确等于 {record['constant']}（共 {record['value_count']} 项"
            + (f"：{extensions}" if extensions else "")
            + "），多一个或少一个文件都会 FAIL"
        ) if record["exact_set"] else record["effect"]
        add_gated(record["scope"], record["gate"], requirement)

    for record in mirrored:
        target = record["tree_a"] if not record["tree_a_outside_repo"] else record["tree_b"]
        if record["tree_a_outside_repo"] and record["tree_b_outside_repo"]:
            continue
        other = record["tree_b"] if target == record["tree_a"] else record["tree_a"]
        add_gated(
            target,
            record["gate"],
            f"编译产物必须与 {other} 保持 {record['comparison']}"
            + ("（另一侧在仓库之外）" if record["tree_b_outside_repo"] else ""),
        )

    for record in diff_constraints:
        for scope in record["compared_scopes"]:
            add_gated(
                scope["scope"],
                record["gate"],
                f"一次只允许 {record['max_changed_files']} 个文件变更："
                + ", ".join(record["allowed_patterns"]),
            )

    for record in textual:
        if record["category"] not in ("count_constraint", "authorization_required"):
            continue
        gate = record["gate_mentions"][0] if record["gate_mentions"] else None
        label = (
            "数量/内容约束" if record["category"] == "count_constraint"
            else "改动需要显式授权或显式流程"
        )
        for subject in record_targets(record):
            add_gated(subject, gate, f"{label}：{record['text']}", record["source"], record["text"])

    gated_keys = {entry["_key"] for entry in gated}
    free_pool = optional_root if presence_enforced else (allow_root - required_root)
    free = []
    for name in sorted(free_pool):
        key = canonical_zone_path(name, root)
        if key in immutable_keys or key in gated_keys or name in forbidden_root:
            continue
        display = directory_display(key, name.endswith("/"))
        if display not in free:
            free.append(display)

    outside_repo = []
    seen_outside = set()
    for record in mirrored:
        for side, flag in (("tree_a", "tree_a_outside_repo"), ("tree_b", "tree_b_outside_repo")):
            if not record[flag]:
                continue
            path_value = record[side]
            if path_value in seen_outside:
                continue
            seen_outside.add(path_value)
            outside_repo.append({
                "path": path_value,
                "relative_to_root": os.path.relpath(path_value, str(root)),
                "requires": "explicit user authorization",
                "why": f"{record['gate']} 把它与仓库内的镜像树做 {record['comparison']} 比对",
                "gate": record["gate"],
            })

    artifacts = []
    artifacts_forbidden = []
    for record in textual:
        if record["category"] != "artifact_location":
            continue
        for subject in record["sentence_subjects"] or record["subjects"]:
            if subject.startswith("/"):
                if subject not in artifacts_forbidden:
                    artifacts_forbidden.append(subject)
                continue
            resolved = resolve_subject(subject, record["file_relative"])
            if not resolved:
                continue
            probe = root / resolved.rstrip("/")
            if probe.exists() and not probe.is_dir():
                continue
            display = as_directory(resolved)
            if display not in artifacts:
                artifacts.append(display)

    artifact_keys = {item.rstrip("/") for item in artifacts}
    normalized_free = []
    for item in free:
        display = as_directory(item) if item.rstrip("/") in artifact_keys else item
        if display not in normalized_free:
            normalized_free.append(display)
    free = normalized_free

    for entry in gated:
        entry.pop("_key", None)

    return {
        "free": free,
        "gated": gated,
        "immutable": immutable,
        "outside_repo": outside_repo,
        "artifacts": artifacts,
        "artifacts_forbidden": artifacts_forbidden,
    }


def build_verify_commands(gate_scripts, contract_text, root, interpreter):
    stem_order = {}
    for entry in gate_scripts:
        stem = Path(entry["path"]).stem
        position = contract_text.find(stem)
        stem_order[stem] = position if position >= 0 else None

    def sort_key(entry):
        stem = Path(entry["path"]).stem
        position = stem_order.get(stem)
        if position is None:
            return (1, entry["requires_gpu"], stem)
        return (0, position, stem)

    ordered = sorted(gate_scripts, key=sort_key)
    commands = []
    index = 0
    for entry in ordered:
        if not entry["runnable"] or not entry["arguments"]:
            continue
        variants = [None]
        for spec in entry["arguments"]:
            if spec["choices"] and not spec["positional"] and len(spec["choices"]) <= 3:
                if spec["dest"] in ("profiler", "mode", "stage", "phase"):
                    variants = list(spec["choices"])
                    break
        for variant in variants:
            index += 1
            parts = [interpreter, entry["path"]]
            placeholders = []
            for spec in entry["arguments"]:
                flag = spec["names"][0]
                if spec["positional"]:
                    token = f"<{spec['dest'].upper()}>"
                    parts.append(token)
                    placeholders.append(token)
                    continue
                if spec["action"] in ("store_true", "store_false"):
                    continue
                is_required = spec["required"] is True
                is_anchor = spec["dest"] in ("root", "workspace", "repo", "repo_root")
                is_variant = variant is not None and spec["choices"] and variant in [
                    str(choice) for choice in spec["choices"]
                ]
                if not (is_required or is_anchor or is_variant):
                    continue
                if is_anchor:
                    parts.extend([flag, str(root)])
                elif is_variant:
                    parts.extend([flag, str(variant)])
                else:
                    token = f"<{spec['dest'].upper()}>"
                    parts.extend([flag, token])
                    placeholders.append(token)
            commands.append({
                "order": index,
                "gate": entry["path"],
                "command": " ".join(parts),
                "placeholders": placeholders,
                "blocking": not entry["requires_gpu"],
                "requires_gpu": entry["requires_gpu"],
                "order_source": "root contract text" if stem_order.get(
                    Path(entry["path"]).stem
                ) is not None else "heuristic (not named in contract)",
                "purpose": entry["docstring"],
            })
    return commands


def regex_arguments(source):
    specs = []
    for match in re.finditer(r"add_argument\s*\(([^)]*)\)", source, re.DOTALL):
        body = match.group(1)
        names = re.findall(r"['\"](-{0,2}[\w\-]+)['\"]", body)
        if not names:
            continue
        head = names[0]
        choices = None
        choices_match = re.search(r"choices\s*=\s*[\(\[\{]([^\)\]\}]*)[\)\]\}]", body)
        if choices_match:
            choices = [
                item.strip().strip("'\"")
                for item in choices_match.group(1).split(",")
                if item.strip()
            ]
        default = None
        default_match = re.search(r"default\s*=\s*([^,\n]+)", body)
        if default_match:
            default = default_match.group(1).strip()
        required = None
        required_match = re.search(r"required\s*=\s*(True|False)", body)
        if required_match:
            required = required_match.group(1) == "True"
        action = None
        action_match = re.search(r"action\s*=\s*['\"]([\w_]+)['\"]", body)
        if action_match:
            action = action_match.group(1)
        type_match = re.search(r"type\s*=\s*([\w.]+)", body)
        specs.append({
            "names": [head],
            "positional": not head.startswith("-"),
            "dest": head.lstrip("-").replace("-", "_"),
            "choices": choices,
            "default": default,
            "default_expr": default,
            "type": type_match.group(1) if type_match else None,
            "required": required,
            "action": action,
            "nargs": None,
            "help": None,
            "line": source[: match.start()].count("\n") + 1,
            "resolved_default": None,
        })
    return specs


def describe_gate(path, root, module, source, reasons, interpreter):
    docstring = None
    runnable = False
    requires_gpu = False
    arguments = []
    parse_error = None
    if module is not None:
        docstring = ast.get_docstring(module.tree)
        for node in module.tree.body:
            if isinstance(node, ast.If):
                test = module.segment(node.test)
                if "__main__" in test:
                    runnable = True
        for node in ast.walk(module.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in ("torch", "cupy", "pycuda"):
                        requires_gpu = True
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] in ("torch", "cupy", "pycuda"):
                    requires_gpu = True
        for spec in module.arg_specs:
            arguments.append({
                key: value for key, value in spec.items() if key != "_default_node"
            })
    else:
        parse_error = "ast.parse failed; fell back to regex"
        arguments = regex_arguments(source)
        runnable = "__main__" in source
        requires_gpu = bool(re.search(r"^\s*import torch|^\s*from torch", source, re.MULTILINE))
    if not runnable and arguments:
        runnable = True
    example = [interpreter, str(path)]
    for spec in arguments:
        if spec["positional"]:
            example.append(f"<{spec['dest'].upper()}>")
        elif spec.get("required") is True:
            example.extend([spec["names"][0], f"<{spec['dest'].upper()}>"])
        elif spec["dest"] in ("root", "workspace", "repo", "repo_root"):
            example.extend([spec["names"][0], str(root)])
    return {
        "path": str(path),
        "relative": path.relative_to(root).as_posix(),
        "detected_by": reasons,
        "docstring": docstring,
        "runnable": runnable,
        "requires_gpu": requires_gpu,
        "parse_error": parse_error,
        "arguments": arguments,
        "invocation_example": " ".join(example),
    }


def render_summary(report):
    zones = report["write_zones"]
    out = []
    out.append("=" * 78)
    out.append("门禁约束探测 (%s)" % report["schema"])
    out.append("root: %s" % report["root"])
    out.append(
        "gate_scripts=%d  path_constraints=%d  mirrored_trees=%d  diff_constraints=%d  textual=%d"
        % (
            len(report["gate_scripts"]),
            len(report["path_constraints"]),
            len(report["mirrored_trees"]),
            len(report["diff_constraints"]),
            len(report["textual_constraints"]),
        )
    )
    out.append("=" * 78)

    out.append("")
    out.append("[1] 要新增文件请放这里")
    if zones["free"]:
        for item in zones["free"]:
            note = "白名单内的可选根条目，可自由新建/修改"
            if item.rstrip("/") in [a.rstrip("/") for a in zones["artifacts"]]:
                note = "临时产物专用目录"
            out.append("    + %-28s %s" % (item, note))
    else:
        out.append("    (未检出可自由写入的区域)")
    if zones["artifacts"]:
        out.append("    临时产物目录: %s" % ", ".join(zones["artifacts"]))
    if zones["artifacts_forbidden"]:
        out.append("    禁止用作产物目录: %s" % ", ".join(zones["artifacts_forbidden"]))

    out.append("")
    out.append("[2] 这些文件碰了必挂门禁")
    for entry in zones["immutable"]:
        out.append("    ! %-28s IMMUTABLE  (%s)" % (entry["path"], entry["source"]))
        out.append("      \"%s\"" % entry["text"])
    for record in report["mirrored_trees"]:
        out.append(
            "    ! %-28s 镜像树 %s"
            % (record["tree_a_relative_to_root"], record["comparison"])
        )
        out.append(
            "      必须与 %s 同步（%s）"
            % (record["tree_b_relative_to_root"], Path(record["gate"]).name)
        )
    for entry in zones["gated"]:
        gate = Path(entry["gate"]).name if entry["gate"] else "-"
        out.append("    ~ %-28s [%s] %s" % (entry["path"], gate, entry["requirement"]))
    for record in report["diff_constraints"]:
        scopes = " vs ".join(item["scope"] for item in record["compared_scopes"]) or "diff 结果"
        out.append(
            "    ~ %-28s [%s] %s"
            % ("diff 范围", Path(record["gate"]).name, scopes)
        )
        out.append("      %s" % record["effect"])

    out.append("")
    out.append("[3] 这些改动需要额外授权")
    printed = False
    for entry in zones["outside_repo"]:
        out.append("    * %s" % entry["path"])
        out.append("      (%s) 需要: %s" % (entry["relative_to_root"], entry["requires"]))
        out.append("      原因: %s" % entry["why"])
        printed = True
    for record in report["textual_constraints"]:
        if record["category"] != "authorization_required":
            continue
        out.append("    * %s  \"%s\"" % (record["source"], record["text"]))
        printed = True
    if not printed:
        out.append("    (未检出需要额外授权的改动)")

    out.append("")
    out.append("[4] 门禁怎么跑")
    if report["verify_commands"]:
        for command in report["verify_commands"]:
            tag = "" if command["blocking"] else "  [非阻塞/需GPU，默认不跑]"
            out.append("    %d) %s%s" % (command["order"], command["command"], tag))
    else:
        out.append("    (未检出可执行门禁)")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect gate scripts and path constraints in a repository."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", default=None)
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--no-default-excludes", action="store_true")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"root is not a directory: {root}", file=sys.stderr)
        return 2

    excludes = build_excludes(args.exclude, not args.no_default_excludes)
    files, skipped = walk_repo(root, excludes)

    interpreter = "python3 -B"
    modules = {}
    gate_scripts = []
    parse_errors = []
    for path in files:
        if path.suffix != ".py":
            continue
        reasons = gate_reasons(root, path)
        if not reasons:
            continue
        source = read_text(path)
        if source is None:
            continue
        module = None
        try:
            module = ModuleAnalysis(path, root, source)
        except SyntaxError as error:
            parse_errors.append({"path": str(path), "error": str(error)})
        except RecursionError as error:
            parse_errors.append({"path": str(path), "error": repr(error)})
        if module is not None:
            modules[path.relative_to(root).as_posix()] = module
        gate_scripts.append(
            describe_gate(path, root, module, source, reasons, interpreter)
        )

    gate_stems = sorted({Path(entry["path"]).stem for entry in gate_scripts})
    path_constraints = extract_path_constraints(modules)
    mirrored_trees = extract_mirrored_trees(modules, root)
    diff_constraints = extract_diff_constraints(modules, root)
    textual_constraints = extract_textual_constraints(files, root, gate_stems)
    contract_blob = "\n".join(
        read_text(path) or ""
        for path in files
        if path.name in CONTRACT_FILENAMES and path.parent == root
    )
    match = re.search(r"python3?\s+-B\b", contract_blob)
    if match:
        interpreter = match.group(0)
    write_zones = build_write_zones(
        root, path_constraints, mirrored_trees, diff_constraints, textual_constraints
    )
    verify_commands = build_verify_commands(
        gate_scripts, contract_blob, root, interpreter
    )

    report = {
        "schema": SCHEMA,
        "root": str(root),
        "detected_at_utc": utc_now(),
        "gate_scripts": gate_scripts,
        "path_constraints": path_constraints,
        "mirrored_trees": mirrored_trees,
        "diff_constraints": diff_constraints,
        "textual_constraints": textual_constraints,
        "write_zones": write_zones,
        "verify_commands": verify_commands,
        "scan": {
            "excludes": excludes,
            "files_scanned": len(files),
            "skipped_directories": skipped,
            "python_gate_modules": sorted(modules),
            "parse_errors": parse_errors,
            "interpreter": interpreter,
        },
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    if args.out and args.out != "-":
        destination = Path(args.out).expanduser()
        if not destination.is_absolute():
            destination = (Path.cwd() / destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    try:
        if args.out == "-" or args.print_json:
            sys.stdout.write(payload)
        if not args.quiet:
            print(render_summary(report))
            if args.out and args.out != "-":
                print("JSON: %s" % destination)
        sys.stdout.flush()
    except BrokenPipeError:
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
