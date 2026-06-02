from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "encoding_audit.csv"
SELF_PATH = Path("scripts/encoding_audit.py")

INCLUDE_DIRS = {"backend", "frontend", "docs", "tests", "plans"}
INCLUDE_ROOT_FILES = {
    ".gitignore",
    ".dockerignore",
    ".env",
    "AGENT.md",
    "CHANGELOG.md",
    "Dockerfile.backend",
    "Dockerfile.frontend",
    "README.md",
    "SECURITY.md",
    "app.py",
    "check_docker_ports.ps1",
    "diagnose_ports.ps1",
    "docker-compose.yml",
    "package.json",
    "plan_tutorial.md",
    "requirements.txt",
    "run_docker.bat",
    "run_local.bat",
    "run_silent.bat",
    "run_silent.vbs",
    "seguimiento_guardado_circulo.md",
    "stop_servers.ps1",
}
EXCLUDED_PARTS = {
    ".git",
    ".codegraph",
    ".kilo",
    ".pytest_cache",
    ".roo",
    "__pycache__",
    "catboost_info",
    "data",
    "dist",
    "img",
    "node_modules",
    "outputs",
    "venv",
}
AUDIT_ARTIFACTS = {
    Path(".editorconfig"),
    Path("encoding_audit.csv"),
    SELF_PATH,
}


@dataclass
class AuditRow:
    path: str
    kind: str
    detected_encoding: str
    has_bom: bool
    is_binary: bool
    status: str
    notes: str


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel in AUDIT_ARTIFACTS:
        return False
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return False
    if len(rel.parts) == 1:
        return rel.name in INCLUDE_ROOT_FILES
    return rel.parts[0] in INCLUDE_DIRS


def detect_encoding(path: Path) -> tuple[str, bool, bool]:
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-bom", True, False
    if data.startswith(b"\xff\xfe\x00\x00"):
        return "utf-32-le", False, False
    if data.startswith(b"\x00\x00\xfe\xff"):
        return "utf-32-be", False, False
    if data.startswith(b"\xff\xfe"):
        return "utf-16-le", False, False
    if data.startswith(b"\xfe\xff"):
        return "utf-16-be", False, False
    if b"\x00" in data[:8192]:
        return "binary-or-unknown", False, True
    try:
        data.decode("utf-8")
        return "utf-8", False, False
    except UnicodeDecodeError:
        pass
    try:
        data.decode("cp1252")
        return "cp1252", False, False
    except UnicodeDecodeError:
        return "binary-or-unknown", False, True


def build_row(path: Path) -> AuditRow:
    rel = path.relative_to(ROOT).as_posix()
    encoding, has_bom, is_binary = detect_encoding(path)
    kind = "binary" if is_binary else "text"
    if is_binary:
        status = "skip_binary"
        notes = "Excluded from text normalization."
    elif encoding == "utf-8":
        status = "ok"
        notes = "Matches repository standard UTF-8 without BOM."
    else:
        status = "needs_conversion"
        notes = "Convert to UTF-8 without BOM."
    return AuditRow(
        path=rel,
        kind=kind,
        detected_encoding=encoding,
        has_bom=has_bom,
        is_binary=is_binary,
        status=status,
        notes=notes,
    )


def iter_scope() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*") if path.is_file() and should_include(path))


def write_csv(rows: list[AuditRow], target: Path) -> None:
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "kind",
                "detected_encoding",
                "has_bom",
                "is_binary",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "path": row.path,
                    "kind": row.kind,
                    "detected_encoding": row.detected_encoding,
                    "has_bom": str(row.has_bom).lower(),
                    "is_binary": str(row.is_binary).lower(),
                    "status": row.status,
                    "notes": row.notes,
                }
            )


def convert_file(path: Path, encoding: str) -> None:
    if encoding == "utf-8-bom":
        text = path.read_text(encoding="utf-8-sig")
    elif encoding == "cp1252":
        text = path.read_text(encoding="cp1252")
    elif encoding.startswith("utf-16") or encoding.startswith("utf-32"):
        text = path.read_text(encoding=encoding)
    else:
        return
    path.write_text(text, encoding="utf-8", newline="\n")


def normalize_scope(rows: list[AuditRow]) -> int:
    changed = 0
    for row in rows:
        if row.status != "needs_conversion":
            continue
        convert_file(ROOT / row.path, row.detected_encoding)
        changed += 1
    return changed


def summarize(rows: list[AuditRow]) -> str:
    counter = Counter(row.detected_encoding for row in rows)
    parts = [f"{key}={counter[key]}" for key in sorted(counter)]
    return f"files={len(rows)} " + " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and normalize repository text encodings.")
    parser.add_argument("--fix", action="store_true", help="Convert files that are not UTF-8 without BOM.")
    parser.add_argument("--csv", default=str(OUTPUT_CSV), help="Path for the generated CSV report.")
    args = parser.parse_args()

    rows = [build_row(path) for path in iter_scope()]
    if args.fix:
        changed = normalize_scope(rows)
        rows = [build_row(path) for path in iter_scope()]
        print(f"normalized_files={changed}")
    write_csv(rows, Path(args.csv))
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
