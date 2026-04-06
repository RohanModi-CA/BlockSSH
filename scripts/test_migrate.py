import msgpack
import math
import numpy as np
from pathlib import Path

def wrap_angle(theta):
    """Wrap angle to [-pi/2, pi/2]"""
    if not math.isfinite(theta):
        return theta
    # atan2(sin(2*theta), cos(2*theta)) / 2
    return 0.5 * math.atan2(math.sin(2.0 * theta), math.cos(2.0 * theta))

def transform_angle(theta_old):
    """
    Transform from horizontal-ref (theta_old=pi/2 is vertical) 
    to vertical-ref (theta_new=0 is vertical).
    Equivalent to rotating by 90 degrees.
    """
    if not math.isfinite(theta_old):
        return theta_old
    
    # In old system: vertical is pi/2 or -pi/2
    # In new system: we want vertical to be 0
    # So we subtract pi/2
    theta_new = theta_old - (math.pi / 2.0)
    return wrap_angle(theta_new)

def inspect_dataset(path):
    p = Path(path)
    t1_path = p / "track1.msgpack"
    with open(t1_path, "rb") as f:
        t1 = msgpack.unpackb(f.read(), raw=False)
    
    angles = []
    for frame in t1["frames"]:
        for det in frame["detections"]:
            if "angle" in det and math.isfinite(det["angle"]):
                angles.append(det["angle"])
    
    if angles:
        print(f"Track1 angles: min={min(angles):.4f}, max={max(angles):.4f}, mean={sum(angles)/len(angles):.4f}")
        print(f"Sample (first 5): {[f'{a:.4f}' for a in angles[:5]]}")
    else:
        print("No angles found in track1.msgpack")

    t2a_path = p / "components" / "a" / "track2_permanence.msgpack"
    if t2a_path.exists():
        with open(t2a_path, "rb") as f:
            t2a = msgpack.unpackb(f.read(), raw=False)
        flat_angles = [a for row in t2a["xPositions"] for a in row if math.isfinite(a)]
        if flat_angles:
            print(f"Track2A angles: min={min(flat_angles):.4f}, max={max(flat_angles):.4f}, mean={sum(flat_angles)/len(flat_angles):.4f}")
            print(f"Sample row 0: {[f'{a:.4f}' for a in t2a['xPositions'][0][:5]]}")

def migrate_dataset(path):
    p = Path(path)
    
    # 1. Migrate track1.msgpack
    t1_path = p / "track1.msgpack"
    with open(t1_path, "rb") as f:
        t1 = msgpack.unpackb(f.read(), raw=False)
    
    for frame in t1["frames"]:
        for det in frame["detections"]:
            if "angle" in det:
                det["angle"] = transform_angle(det["angle"])
    
    with open(t1_path, "wb") as f:
        f.write(msgpack.packb(t1, use_bin_type=True))
    print(f"Migrated {t1_path}")

    # 2. Migrate track2_permanence.msgpack for 'a' component
    t2a_path = p / "components" / "a" / "track2_permanence.msgpack"
    if t2a_path.exists():
        with open(t2a_path, "rb") as f:
            t2a = msgpack.unpackb(f.read(), raw=False)
        
        new_pos = []
        for row in t2a["xPositions"]:
            new_pos.append([transform_angle(a) for a in row])
        t2a["xPositions"] = new_pos
        
        with open(t2a_path, "wb") as f:
            f.write(msgpack.packb(t2a, use_bin_type=True))
        print(f"Migrated {t2a_path}")

if __name__ == "__main__":
    import sys
    test_path = "track/data/MIGRATE_TEST"
    print("=== BEFORE MIGRATION ===")
    inspect_dataset(test_path)
    print("\nApplying migration...")
    migrate_dataset(test_path)
    print("\n=== AFTER MIGRATION ===")
    inspect_dataset(test_path)
