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


def _match_name(name: str, token: str) -> bool:
    token = Path(token).stem
    return token == name or name.endswith(token)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Bottom/1.TrackRun.py on multiple videos.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Video names or numeric suffixes to exclude.")
    parser.add_argument("--nojsons", action="store_true", help="Process videos even if params are missing.")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview during tracking.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    videos = find_videos()
    if not videos:
        print("No videos found in track/Videos/.")
        return 0

    tasks: list[str] = []
    for video_path in videos:
        name = video_name(video_path)
        if any(_match_name(name, token) for token in args.exclude):
            continue
        has_params = params_bottom_path(name).exists() or legacy_params_black_path(name).exists()
        if not args.nojsons and not has_params:
            continue
        tasks.append(name)

    if not tasks:
        print("No videos selected for bottom tracking.")
        return 0

    print(f"{len(tasks)} video(s) will be processed:\n")
    for name in tasks:
        has_params = params_bottom_path(name).exists() or legacy_params_black_path(name).exists()
        status = "params available" if has_params else "no params"
        print(f"  {name}  ({status})")
    print()

    for idx, name in enumerate(tasks, start=1):
        print(f"[{idx}/{len(tasks)}] Tracking {name}...\n")
        cmd = ["python3", "track/Bottom/1.TrackRun.py", name]
        if args.no_preview:
            cmd.append("--no-preview")
        subprocess.run(cmd)
        print()

    print("Bottom batch tracking complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
