#!/usr/bin/env python3
"""
Move images with hyphens in their filenames from data/raw/dataset/{BookID}/characters/{Unicode}
into data/raw/error, preserving the relative directory structure.

Usage:
  python scripts/move_hyphen_images.py [--root data/raw] [--dry-run] [--include-ext EXT ...] [--exclude-ext EXT ...]

Defaults:
  - root: data/raw
  - include-ext: common image extensions (jpg, jpeg, png, webp, bmp, tiff, tif, gif)

Notes:
  - Files are moved if their basename contains a hyphen '-'.
  - The target path will be: {root}/error/{relative_path_from_dataset}
  - The relative path from dataset is everything under dataset/, e.g., dataset/123/characters/4E00/img-001.jpg
    becomes error/123/characters/4E00/img-001.jpg
  - If the destination file already exists, a numeric suffix is appended before the extension.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Set

COMMON_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}


def iter_image_files(base: Path, include_ext: Set[str], exclude_ext: Set[str]) -> Iterable[Path]:
    # Walk only under dataset/*/characters/* directories
    dataset_dir = base / "dataset"
    if not dataset_dir.exists():
        return []

    # Pattern: data/raw/dataset/{BookID}/characters/{Unicode}/**/*
    for book_dir in dataset_dir.iterdir():
        if not book_dir.is_dir():
            continue
        chars_dir = book_dir / "characters"
        if not chars_dir.is_dir():
            continue
        for unicode_dir in chars_dir.iterdir():
            if not unicode_dir.is_dir():
                continue
            for p in unicode_dir.rglob("*"):
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if include_ext and ext not in include_ext:
                    continue
                if exclude_ext and ext in exclude_ext:
                    continue
                yield p


def ensure_unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        candidate = parent / f"{stem}({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def move_hyphen_images(root: Path, dry_run: bool, include_ext: Set[str], exclude_ext: Set[str]) -> int:
    moved = 0
    base = root
    error_root = base / "error"
    dataset_dir = base / "dataset"

    if not dataset_dir.exists():
        print(f"[INFO] No dataset directory found at {dataset_dir}. Nothing to do.")
        return 0

    for src in iter_image_files(base, include_ext, exclude_ext):
        if "-" not in src.name:
            continue
        # Compute relative path from dataset dir
        rel = src.relative_to(dataset_dir)
        dst = error_root / rel
        if dry_run:
            print(f"[DRY-RUN] Would move: {src} -> {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            final_dst = ensure_unique_path(dst)
            print(f"[MOVE] {src} -> {final_dst}")
            shutil.move(str(src), str(final_dst))
        moved += 1

    if moved == 0:
        print("[INFO] No images with hyphens found.")
    else:
        print(f"[INFO] Done. {'Would move' if dry_run else 'Moved'} {moved} file(s).")
    return moved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Move images with '-' in filename to error directory")
    p.add_argument("--root", type=Path, default=Path("data/raw"), help="Root directory (default: data/raw)")
    p.add_argument("--dry-run", action="store_true", help="Show actions without moving files")
    p.add_argument(
        "--include-ext",
        nargs="*",
        default=None,
        help="Whitelist of file extensions to consider (e.g., jpg png). Defaults to common image types.",
    )
    p.add_argument(
        "--exclude-ext",
        nargs="*",
        default=None,
        help="Blacklist of file extensions to skip (e.g., json txt)",
    )
    return p.parse_args()


def normalize_exts(exts: list[str] | None) -> Set[str]:
    if not exts:
        return set()
    normed = set()
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        normed.add(e)
    return normed


def main():
    args = parse_args()
    include_ext = normalize_exts(args.include_ext) or COMMON_IMAGE_EXTS
    exclude_ext = normalize_exts(args.exclude_ext)

    move_hyphen_images(args.root, args.dry_run, include_ext, exclude_ext)


if __name__ == "__main__":
    main()
