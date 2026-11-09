"""Shared file operations and UTC conversions at product boundaries."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256(path: Path | str) -> str:
    """Hash a file in bounded chunks without changing its byte-level identity."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str, *, encoding: str | None = None) -> Any:
    return json.loads(Path(path).read_text(encoding=encoding))


def read_json_object(path: Path | str) -> dict[str, Any]:
    """Read an object-valued UTF-8 product, rejecting other JSON values."""
    path = Path(path)
    payload = read_json(path, encoding="utf-8")
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def display_path(path: Path | str, *, root: Path | str) -> str:
    """Use a path relative to the supplied root when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def utc_timestamp(value) -> pd.Timestamp:
    """Interpret naive values as UTC and convert aware values to UTC."""
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def utc_text(value) -> str:
    return utc_timestamp(value).isoformat().replace("+00:00", "Z")


@contextmanager
def atomic_output_paths(
    paths: Iterable[Path | str], *, keep_suffix: bool = False,
    temporary_paths: Iterable[Path | str] | None = None,
) -> Iterator[tuple[Path, ...]]:
    """Stage every output before replacing destinations in their supplied order.

    Each replacement is atomic. The group has the same sequential replacement
    behavior as the builders, so it is not a multi-file transaction. Keeping the
    final suffix supports writers such as NumPy that infer a file type from it.
    """
    paths = tuple(Path(path) for path in paths)
    if temporary_paths is None:
        temporary = tuple(
            path.with_name(path.stem + ".tmp" + path.suffix)
            if keep_suffix else path.with_suffix(path.suffix + ".tmp")
            for path in paths
        )
    else:
        temporary = tuple(Path(path) for path in temporary_paths)
        if len(temporary) != len(paths):
            raise ValueError("each atomic output requires one temporary path")
    yield temporary
    for source, target in zip(temporary, paths):
        source.replace(target)
