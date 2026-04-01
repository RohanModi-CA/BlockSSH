import msgpack
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_msgpack(path):
    with open(path, "rb") as f:
        return msgpack.unpackb(f.read(), raw=False)

def plot_migration_comparison(dataset_old_path, dataset_new_path):
    # 1. Load OLD track1
    old_t1 = load_msgpack(Path(dataset_old_path) / "track1.msgpack")
    old_times = []
    old_angles = []
    for f in old_t1["frames"]:
        for d in f["detections"]:
            if "angle" in d and math.isfinite(d["angle"]):
                old_times.append(f["frame_time_s"])
                old_angles.append(d["angle"])
    
    # 2. Load NEW track1 (from MIGRATE_TEST)
    new_t1 = load_msgpack(Path(dataset_new_path) / "track1.msgpack")
    new_times = []
    new_angles = []
    for f in new_t1["frames"]:
        for d in f["detections"]:
            if "angle" in d and math.isfinite(d["angle"]):
                new_times.append(f["frame_time_s"])
                new_angles.append(d["angle"])

    # 3. Load track2 angle permanence (A component)
    # Note: track2_a stores cos(2*theta)
    old_t2a_path = Path(dataset_old_path) / "components/a/track2_permanence.msgpack"
    new_t2a_path = Path(dataset_new_path) / "components/a/track2_permanence.msgpack"
    
    old_t2a = load_msgpack(old_t2a_path) if old_t2a_path.exists() else None
    new_t2a = load_msgpack(new_t2a_path) if new_t2a_path.exists() else None

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: Raw Track1 Angles (Horizontal Ref vs Vertical Ref)
    axes[0].scatter(old_times, old_angles, s=2, alpha=0.5, label="Old (Horizontal Ref)")
    axes[0].scatter(new_times, new_angles, s=2, alpha=0.5, label="New (Vertical Ref)")
    axes[0].set_ylabel("Angle (rad)")
    axes[0].set_title("Track1 Raw Angles Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Old Track2 A component (cos(2*theta_old))
    if old_t2a:
        t2_times = old_t2a["frameTimes_s"]
        t2_pos = np.array(old_t2a["xPositions"])
        for col in range(t2_pos.shape[1]):
            axes[1].plot(t2_times, t2_pos[:, col], label=f"Block {col}" if col < 5 else None)
        axes[1].set_ylabel("cos(2*theta)")
        axes[1].set_title("Old Track2 A component (cos(2*theta_old))")
        axes[1].grid(True, alpha=0.3)

    # Subplot 3: New Track2 A component (cos(2*theta_new))
    if new_t2a:
        t2_times = new_t2a["frameTimes_s"]
        t2_pos = np.array(new_t2a["xPositions"])
        for col in range(t2_pos.shape[1]):
            axes[2].plot(t2_times, t2_pos[:, col], label=f"Block {col}" if col < 5 else None)
        axes[2].set_ylabel("cos(2*theta)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("New Track2 A component (cos(2*theta_new))")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    print("Saving comparison plot to 'migration_comparison.png' and showing...")
    plt.savefig("migration_comparison.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_migration_comparison("track/data/IMG_0723", "track/data/MIGRATE_TEST")
