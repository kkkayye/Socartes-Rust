#!/usr/bin/env python3
"""Lightweight API parity scanner for Socartes-Rust replacement work.

Scans three sources and prints JSON to stdout:
- Rust backend routes from backend/src/lib.rs
- DeepTutor web frontend apiUrl/wsUrl usages
- DeepTutor Python FastAPI routers and include_router prefixes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
JS_FILE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx"}

RUST_ROUTE_RE = re.compile(
    r"""\.route\(\s*"(?P<path>[^"]+)"\s*,\s*(?P<method>get|post|put|patch|delete)\s*\(""",
    re.MULTILINE,
)
RUST_CHAINED_METHOD_RE = re.compile(r"""\.(?P<method>get|post|put|patch|delete)\s*\(""")

PY_INCLUDE_ROUTER_RE = re.compile(
    r"""app\.include_router\(\s*(?P<module>\w+)\.router\s*,\s*prefix\s*=\s*(?P<prefix>["'][^"']*["'])""",
    re.MULTILINE,
)

PY_ROUTER_DECORATOR_RE = re.compile(
    r"""@router\.(?P<method>get|post|put|delete|patch|websocket)\(\s*(?P<path>["'][^"']*["'])""",
    re.MULTILINE,
)

JS_LITERAL_CONST_RE = re.compile(
    r"""(?:const|let|var)\s+(?P<name>[A-Z][A-Z0-9_]*)\s*=\s*(?P<value>"[^"]*"|'[^']*'|`[^`]*`)\s*;""",
    re.MULTILINE,
)

JS_API_CALL_RE = re.compile(
    r"""(?P<kind>apiUrl|wsUrl)\(\s*(?P<arg>"[^"]*"|'[^']*'|`[^`]*`)\s*\)""",
    re.MULTILINE,
)

JS_FETCH_API_CALL_RE = re.compile(
    r"""fetch\(\s*apiUrl\(\s*(?P<arg>"[^"]*"|'[^']*'|`[^`]*`)\s*\)\s*(?:,\s*(?P<options>\{.*?\}))?\s*\)""",
    re.DOTALL,
)

JS_METHOD_RE = re.compile(r"""method\s*:\s*["'](?P<method>[A-Za-z]+)["']""")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rust-root",
        required=True,
        help="Path to the Socartes-Rust repository root.",
    )
    parser.add_argument(
        "--deeptutor-root",
        required=True,
        help="Path to the DeepTutor repository root.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation size. Default: 2",
    )
    return parser.parse_args()


def normalize_path(path: str) -> str:
    if not path:
        return "/"
    if not path.startswith("/"):
        path = "/" + path
    return re.sub(r"/{2,}", "/", path)


def canonical_path(path: str) -> str:
    """Normalize dynamic segments so equivalent routes compare cleanly."""
    normalized = normalize_path(path).split("?", 1)[0]
    normalized = re.sub(r"\{\*[^}]+\}", "{path}", normalized)
    normalized = re.sub(r"\{[^}]+\}", "{param}", normalized)
    normalized = re.sub(r":path\b", "", normalized)
    return normalized


def quote_text(text: str) -> str:
    return text[1:-1]


def render_js_template(text: str, constants: dict[str, str]) -> str:
    if text.startswith(("'", '"')):
        return quote_text(text)
    body = quote_text(text)
    for name, value in constants.items():
        body = body.replace("${" + name + "}", value)
    body = re.sub(r"\$\{[^}]+\}", "{expr}", body)
    return body


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def scan_rust_routes(rust_root: Path) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    lib_path = rust_root / "backend" / "src" / "lib.rs"
    routes: list[dict] = []
    if not lib_path.exists():
        errors.append(f"missing Rust route file: {lib_path}")
        return routes, errors

    text = read_text(lib_path)
    matches = list(RUST_ROUTE_RE.finditer(text))
    for index, match in enumerate(matches):
        path = normalize_path(match.group("path"))
        methods = {match.group("method").upper()}
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        route_snippet = text[match.start() : next_start]
        methods.update(
            chained.group("method").upper()
            for chained in RUST_CHAINED_METHOD_RE.finditer(route_snippet)
        )
        for method in sorted(methods):
            routes.append(
                {
                    "method": method,
                    "path": path,
                    "handler": None,
                    "source": {"file": str(lib_path), "line": line_number(text, match.start())},
                }
            )
    return routes, errors


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def parse_python_router_prefixes(main_text: str) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for match in PY_INCLUDE_ROUTER_RE.finditer(main_text):
        prefixes[match.group("module")] = normalize_path(quote_text(match.group("prefix")))
    return prefixes


def join_paths(prefix: str, path: str) -> str:
    prefix = normalize_path(prefix)
    route = quote_text(path) if path.startswith(("'", '"')) else path
    if not route:
        route = "/"
    if route == "":
        route = "/"
    if route == "/":
        return prefix
    return normalize_path(prefix.rstrip("/") + "/" + route.lstrip("/"))


def scan_python_routes(deeptutor_root: Path) -> tuple[list[dict], list[str], dict[str, str]]:
    errors: list[str] = []
    main_path = deeptutor_root / "deeptutor" / "api" / "main.py"
    routers_dir = deeptutor_root / "deeptutor" / "api" / "routers"
    routes: list[dict] = []
    prefixes: dict[str, str] = {}

    if not main_path.exists():
        errors.append(f"missing DeepTutor API main file: {main_path}")
        return routes, errors, prefixes
    if not routers_dir.exists():
        errors.append(f"missing DeepTutor routers directory: {routers_dir}")
        return routes, errors, prefixes

    main_text = read_text(main_path)
    prefixes = parse_python_router_prefixes(main_text)

    for router_file in sorted(routers_dir.glob("*.py")):
        if router_file.name == "__init__.py":
            continue
        module_name = router_file.stem
        prefix = prefixes.get(module_name, "")
        text = read_text(router_file)
        for match in PY_ROUTER_DECORATOR_RE.finditer(text):
            method = match.group("method").upper()
            full_path = join_paths(prefix or "/", match.group("path"))
            routes.append(
                {
                    "method": method,
                    "path": full_path,
                    "module": module_name,
                    "source": {
                        "file": str(router_file),
                        "line": line_number(text, match.start()),
                    },
                }
            )
    return routes, errors, prefixes


def iter_frontend_files(web_root: Path) -> Iterable[Path]:
    preferred_dirs = ("app", "components", "lib")
    for name in preferred_dirs:
        base = web_root / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix in JS_FILE_SUFFIXES and path.is_file():
                yield path


def extract_js_constants(text: str) -> dict[str, str]:
    constants: dict[str, str] = {}
    for match in JS_LITERAL_CONST_RE.finditer(text):
        raw_value = match.group("value")
        value = render_js_template(raw_value, constants)
        constants[match.group("name")] = value
    return constants


def infer_fetch_method(options: str | None) -> str:
    if not options:
        return "GET"
    match = JS_METHOD_RE.search(options)
    if not match:
        return "GET"
    return match.group("method").upper()


def scan_frontend_calls(deeptutor_root: Path) -> tuple[list[dict], list[str]]:
    errors: list[str] = []
    web_root = deeptutor_root / "web"
    calls: list[dict] = []
    if not web_root.exists():
        errors.append(f"missing DeepTutor web root: {web_root}")
        return calls, errors

    for file_path in sorted(iter_frontend_files(web_root)):
        text = read_text(file_path)
        constants = extract_js_constants(text)
        seen: set[tuple[str, int, str]] = set()

        for match in JS_FETCH_API_CALL_RE.finditer(text):
            raw_path = render_js_template(match.group("arg"), constants)
            method = infer_fetch_method(match.group("options"))
            line = line_number(text, match.start())
            key = ("fetch", line, raw_path)
            if key in seen:
                continue
            seen.add(key)
            calls.append(
                {
                    "kind": "fetch",
                    "transport": "http",
                    "method": method,
                    "path": normalize_path(raw_path),
                    "source": {"file": str(file_path), "line": line},
                }
            )

        for match in JS_API_CALL_RE.finditer(text):
            kind = match.group("kind")
            raw_path = render_js_template(match.group("arg"), constants)
            line = line_number(text, match.start())
            key = (kind, line, raw_path)
            if key in seen:
                continue
            seen.add(key)
            calls.append(
                {
                    "kind": kind,
                    "transport": "ws" if kind == "wsUrl" else "http",
                    "method": "WS" if kind == "wsUrl" else None,
                    "path": normalize_path(raw_path),
                    "source": {"file": str(file_path), "line": line},
                }
            )
    return calls, errors


def unique_paths(entries: Iterable[dict], *, methods: set[str] | None = None) -> set[str]:
    result = set()
    for entry in entries:
        method = entry.get("method")
        if methods is not None and method not in methods:
            continue
        result.add(canonical_path(entry["path"]))
    return result


def count_methods(entries: Iterable[dict]) -> dict[str, int]:
    counter = Counter()
    for entry in entries:
        method = entry.get("method") or "UNKNOWN"
        counter[method] += 1
    return dict(sorted(counter.items()))


def build_summary(rust_routes: list[dict], python_routes: list[dict], frontend_calls: list[dict]) -> dict:
    rust_http_paths = unique_paths(rust_routes)
    python_http_paths = unique_paths(
        python_routes,
        methods=set(HTTP_METHODS) | {"PATCH"},
    )
    python_ws_paths = unique_paths(python_routes, methods={"WEBSOCKET"})
    frontend_http_paths = unique_paths(
        (call for call in frontend_calls if call["transport"] == "http")
    )
    frontend_ws_paths = unique_paths(
        (call for call in frontend_calls if call["transport"] == "ws")
    )

    return {
        "counts": {
            "rust_routes": len(rust_routes),
            "python_routes": len(python_routes),
            "frontend_calls": len(frontend_calls),
        },
        "methods": {
            "rust": count_methods(rust_routes),
            "python": count_methods(python_routes),
            "frontend": count_methods(frontend_calls),
        },
        "path_sets": {
            "rust_http_paths": sorted(rust_http_paths),
            "python_http_paths": sorted(python_http_paths),
            "python_ws_paths": sorted(python_ws_paths),
            "frontend_http_paths": sorted(frontend_http_paths),
            "frontend_ws_paths": sorted(frontend_ws_paths),
        },
        "parity": {
            "rust_vs_python_missing_in_rust": sorted(python_http_paths - rust_http_paths),
            "rust_vs_frontend_missing_in_rust": sorted(frontend_http_paths - rust_http_paths),
            "rust_only_http_paths": sorted(rust_http_paths - python_http_paths),
            "frontend_ws_missing_in_rust": sorted(frontend_ws_paths - rust_http_paths),
            "frontend_ws_missing_in_python_ws": sorted(frontend_ws_paths - python_ws_paths),
            "shared_http_paths_all_three": sorted(
                rust_http_paths & python_http_paths & frontend_http_paths
            ),
        },
    }


def build_report(rust_root: Path, deeptutor_root: Path) -> dict:
    rust_routes, rust_errors = scan_rust_routes(rust_root)
    python_routes, python_errors, router_prefixes = scan_python_routes(deeptutor_root)
    frontend_calls, frontend_errors = scan_frontend_calls(deeptutor_root)

    report = {
        "meta": {
            "rust_root": str(rust_root),
            "deeptutor_root": str(deeptutor_root),
            "scanner": "tools/api_parity_scan.py",
        },
        "rust": {
            "route_file": str(rust_root / "backend" / "src" / "lib.rs"),
            "routes": rust_routes,
            "errors": rust_errors,
        },
        "python": {
            "main_file": str(deeptutor_root / "deeptutor" / "api" / "main.py"),
            "router_prefixes": router_prefixes,
            "routes": python_routes,
            "errors": python_errors,
        },
        "frontend": {
            "web_root": str(deeptutor_root / "web"),
            "calls": frontend_calls,
            "errors": frontend_errors,
        },
    }
    report["summary"] = build_summary(rust_routes, python_routes, frontend_calls)
    return report


def main() -> int:
    args = parse_args()
    rust_root = Path(args.rust_root).expanduser().resolve()
    deeptutor_root = Path(args.deeptutor_root).expanduser().resolve()

    report = build_report(rust_root, deeptutor_root)
    json.dump(report, sys.stdout, indent=args.indent, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
