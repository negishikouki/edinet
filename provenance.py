from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(path: Path, content: str) -> None:
    write_bytes_atomic(path, content.encode("utf-8"))


def file_identity(path: Path, *, root: Path | None = None) -> dict[str, object]:
    resolved = path.resolve()
    identity_path = (
        resolved.relative_to(root.resolve()).as_posix()
        if root is not None
        else str(resolved)
    )
    return {
        "path": identity_path,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
