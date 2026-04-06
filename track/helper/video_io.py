"""
helper/video_io.py
Utilities for locating video files and resolving dataset paths.
"""

import os
from typing import Optional

VIDEO_EXTENSIONS = ['.mov', '.avi', '.mp4']
VIDEO_DIR = "Videos"
DATA_DIR  = "data"


def find_video(name_or_prefix: str, search_dir: str = VIDEO_DIR) -> Optional[str]:
    """
    Search search_dir for a video whose base name matches name_or_prefix.

    Matching strategy (in order):
      1. Exact base name + each extension (.mov, .avi, .mp4), case-insensitive.
      2. Suffix match: any file whose base name ends with name_or_prefix
         (e.g. '9282' matches 'IMG_9282').

    Returns the full path to the first match, or None if not found.
    """
    if not os.path.isdir(search_dir):
        raise FileNotFoundError(f"Video directory not found: '{search_dir}'")

    base     = os.path.splitext(name_or_prefix)[0]
    ext_set  = {e.lower() for e in VIDEO_EXTENSIONS}

    # 1. Exact match
    for ext in VIDEO_EXTENSIONS:
        for variant in (ext, ext.upper(), ext.capitalize()):
            path = os.path.join(search_dir, base + variant)
            if os.path.isfile(path):
                return path

    # 2. Suffix match (sorted for determinism)
    for fname in sorted(os.listdir(search_dir)):
        fbase, fext = os.path.splitext(fname)
        if fext.lower() in ext_set:
            if fbase == base or fbase.endswith(base):
                return os.path.join(search_dir, fname)

    return None


def video_name(video_path: str) -> str:
    """Return the base filename without extension."""
    return os.path.splitext(os.path.basename(video_path))[0]


def dataset_dir(name: str) -> str:
    """Return data/{name}/  (name may include path or extension; both are stripped)."""
    clean = os.path.splitext(os.path.basename(name))[0]
    return os.path.join(DATA_DIR, clean)


def ensure_dataset_dir(name: str) -> str:
    """Create and return the dataset directory."""
    d = dataset_dir(name)
    os.makedirs(d, exist_ok=True)
    return d


def params_path(name: str) -> str:
    return os.path.join(dataset_dir(name), "params.json")


def track1_output_path(name: str) -> str:
    return os.path.join(dataset_dir(name), "track1.msgpack")


def track2_output_path(name: str) -> str:
    return os.path.join(dataset_dir(name), "track2_permanence.msgpack")


def component_dir(name: str, component: str) -> str:
    return os.path.join(dataset_dir(name), "components", str(component))


def component_track2_output_path(name: str, component: str) -> str:
    return os.path.join(component_dir(name, component), "track2_permanence.msgpack")
