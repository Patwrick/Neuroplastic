"""Verify the selected release bytes without executing model/reference code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Docs/research/csgn_v0_4_1"

def main():
    manifest = json.loads((PACKAGE / "WORKING_SUBSET_MANIFEST.sha256.json").read_text(encoding="utf-8"))
    errors = []
    expected = set(manifest["local_supplemental_files"])
    for entry in manifest["files"]:
        path = PACKAGE / entry["path"]
        if not path.resolve().is_relative_to(PACKAGE.resolve()):
            errors.append("Unsafe path")
            continue
        expected.add(entry["path"])
        if not path.is_file():
            errors.append("Missing: " + entry["path"])
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            errors.append("Changed: " + entry["path"])
    actual = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file()}
    errors.extend("Unexpected: " + p for p in sorted(actual - expected))
    for entry in manifest["omissions"]:
        if (PACKAGE / entry["path"]).exists():
            errors.append("Omitted history unexpectedly present: " + entry["path"])
    print(json.dumps({"status": "fail" if errors else "pass", "scope": "working subset integrity only", "files_checked": len(manifest["files"]), "original_release_completeness": False, "errors": errors}, indent=2))
    raise SystemExit(bool(errors))

if __name__ == "__main__":
    main()
