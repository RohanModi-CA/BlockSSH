import msgpack
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_msgpack(path):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)

def plot_raw_angle_comparison(old_path, new_path):
    # 1. Load OLD track1
    old_t1 = load_msgpack(Path(old_path) / "track1.msgpack")
    
    # Extract times and angles for ONE block (e.g., the first one in each frame)
    # to see the temporal continuity clearly.
    old_times = []
    old_angles = []
    for f in old_t1["frames"]:
        if f["detections"]:
            # We take the first detection that has a finite angle
            found = False
            for d in f["detections"]:
                if "angle" in d and math.isfinite(d["angle"]):
                    old_times.append(f["frame_time_s"])
                    old_angles.append(d["angle"])
                    found = True
                    break
            if not found:
                old_times.append(f["frame_time_s"])
                old_angles.append(np.nan)
    
    # 2. Load NEW track1 (from MIGRATE_TEST)
    new_t1 = load_msgpack(Path(new_path) / "track1.msgpack")
    new_times = []
    new_angles = []
    for f in new_t1["frames"]:
        if f["detections"]:
            found = False
            for d in f["detections"]:
                if "angle" in d and math.isfinite(d["angle"]):
                    new_times.append(f["frame_time_s"])
                    new_angles.append(d["angle"])
                    found = True
                    break
            if not found:
                new_times.append(f["frame_time_s"])
                new_angles.append(np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Subplot 1: Old Raw Track1 Angles (Horizontal Ref)
    axes[0].plot(old_times, old_angles, 'o', markersize=1, alpha=0.6, label="Old Raw Detections")
    axes[0].axhline(y=math.pi/2, color='r', linestyle='--', alpha=0.5, label="+pi/2 (Vertical)")
    axes[0].axhline(y=-math.pi/2, color='r', linestyle='--', alpha=0.5, label="-pi/2 (Vertical)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].set_ylim(-2, 2)
    axes[0].set_title("OLD Track1: Raw Angle vs Time (Horizontal = 0, Vertical = +/- pi/2)")
    axes[0].legend(loc='upper right', markerscale=5)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: New Raw Track1 Angles (Vertical Ref)
    axes[1].plot(new_times, new_angles, 'o', color='green', markersize=1, alpha=0.6, label="New Raw Detections")
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label="0 (Vertical)")
    axes[1].set_ylabel("Angle (rad)")
    axes[1].set_ylim(-1, 1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("NEW Track1: Raw Angle vs Time (Vertical = 0)")
    axes[1].legend(loc='upper right', markerscale=5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = "raw_angle_comparison_v2.png"
    print(f"Saving comparison plot to {out_file}...")
    plt.savefig(out_file, dpi=200)
    plt.show()

if __name__ == "__main__":
    plot_raw_angle_comparison("track/data/IMG_0723", "track/data/MIGRATE_TEST")
