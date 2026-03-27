#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from track.core.io import video_name
from track.core.layout import VIDEOS_DIR, legacy_params_black_path, params_bottom_path


VIDEO_EXTS = (".mov", ".avi", ".mp4")


def find_videos(video_dir: Path = VIDEOS_DIR) -> list[Path]:
    vids: list[Path] = []
    for path in sorted(video_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            vids.append(path)
    return vids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Bottom/0.VideoPrepareBottom.py on any video missing bottom params.")
    parser.add_argument("--no-preview", action="store_true", help="Disable detection preview during tests.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    videos = find_videos()
    if not videos:
        print("No videos found in track/Videos/.")
        return 0

    to_prepare: list[str] = []
    for video_path in videos:
        name = video_name(video_path)
        if params_bottom_path(name).exists() or legacy_params_black_path(name).exists():
            continue
        to_prepare.append(name)

    if not to_prepare:
        print("All videos already have bottom-tracking params.")
        return 0

    print(f"{len(to_prepare)} video(s) require preparation:\n")
    for name in to_prepare:
        print(f"  {name}")
    print()

    for idx, name in enumerate(to_prepare, start=1):
        print(f"[{idx}/{len(to_prepare)}] Preparing {name}...\n")
        cmd = ["python3", "track/Bottom/0.VideoPrepareBottom.py", name]
        if args.no_preview:
            cmd.append("--no-preview")
        subprocess.run(cmd)
        print()

    print("Bottom batch preparation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
