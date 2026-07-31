from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

from .progress import ProgressManager, default_progress


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(root: Path, progress: ProgressManager | None = None) -> dict:
    root = root.resolve()
    manager = progress or default_progress()
    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "freeze_manifest.json"
    ]
    files = []
    for path in manager.bar(
        paths,
        total=len(paths),
        desc="Freeze: hashing artifacts",
        unit="file",
        leave=progress is None,
    ):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema": "codegap.freeze.v1",
        "root": str(root),
        "files": files,
    }
    (root / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def verify_freeze(
    manifest_path: Path,
    progress: ProgressManager | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(manifest["root"])
    manager = progress or default_progress()
    errors = []
    entries = manifest["files"]
    for entry in manager.bar(
        entries,
        total=len(entries),
        desc="Freeze: verifying artifacts",
        unit="file",
        leave=progress is None,
    ):
        path = root / entry["path"]
        if not path.is_file():
            errors.append({"path": entry["path"], "error": "missing"})
            continue
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            errors.append(
                {
                    "path": entry["path"],
                    "error": "sha256_mismatch",
                    "expected": entry["sha256"],
                    "actual": actual,
                }
            )
    return {"ok": not errors, "errors": errors}
