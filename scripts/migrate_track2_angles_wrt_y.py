import msgpack
import math
import numpy as np
from pathlib import Path

def wrap_angle(theta):
    """Wrap angle to [-pi/2, pi/2]"""
    if not math.isfinite(theta):
        return theta
    return 0.5 * math.atan2(math.sin(2.0 * theta), math.cos(2.0 * theta))

def migrate_dataset(path: Path):
    t2a_path = path / "components" / "a" / "track2_permanence.msgpack"
    t2x_path = path / "components" / "x" / "track2_permanence.msgpack"
    t1_path = path / "track1.msgpack"
    
    if not t2a_path.exists() or not t2x_path.exists():
        return

    with open(t2a_path, "rb") as f:
        t2a = msgpack.unpackb(f.read(), raw=False)
    with open(t2x_path, "rb") as f:
        t2x = msgpack.unpackb(f.read(), raw=False)
        
    t1_frames = None
    if t1_path.exists():
        with open(t1_path, "rb") as f:
            t1 = msgpack.unpackb(f.read(), raw=False)
            t1_frames = t1.get("frames", [])

    flat_a = [a for row in t2a["xPositions"] for a in row if math.isfinite(a)]
    if not flat_a:
        return
        
    min_a = min(flat_a)
    max_a = max(flat_a)
    
    # If values are outside [-1.01, 1.01], they are definitely already raw angles
    if min_a < -1.01 or max_a > 1.01:
        print(f"Skipping {path.name}: already appears to be raw angles (min={min_a:.2f}, max={max_a:.2f})")
        return
        
    # We need to migrate from cos(2*theta) to theta
    print(f"Migrating {path.name}... (cos2 values found: min={min_a:.2f}, max={max_a:.2f})")
    
    new_pos = []
    n_frames = len(t2a["xPositions"])
    
    # Pre-build track1 lookup if available
    track1_lookup = []
    if t1_frames:
        for f in t1_frames:
            # Map of x to angle
            f_map = {det["x"]: det.get("angle", float("nan")) for det in f.get("detections", [])}
            track1_lookup.append(f_map)
            
    for k in range(n_frames):
        row_a = t2a["xPositions"][k]
        row_x = t2x["xPositions"][k]
        new_row_a = []
        
        t1_map = track1_lookup[k] if track1_lookup and k < len(track1_lookup) else {}
        
        for i in range(len(row_a)):
            val = row_a[i]
            x_val = row_x[i]
            
            if not math.isfinite(val):
                new_row_a.append(val)
                continue
                
            # Fallback magnitude from cos2
            val_clamped = max(-1.0, min(1.0, val))
            mag = 0.5 * math.acos(val_clamped) # in [0, pi/2]
            
            # Try to find exact match in track1
            best_angle = None
            if t1_map and math.isfinite(x_val):
                # find closest x in track1
                xs = list(t1_map.keys())
                if xs:
                    closest_x = min(xs, key=lambda x: abs(x - x_val))
                    if abs(closest_x - x_val) < 0.5:
                        best_angle = t1_map[closest_x]
            
            if best_angle is not None and math.isfinite(best_angle):
                final_angle = best_angle
            else:
                # If no track1 (e.g. synthetic or interpolated), we just use the positive magnitude
                # But we can try to guess the sign from neighboring frames if needed.
                # For now, just use mag. The user requested to "just look at the angle wrt y axis".
                final_angle = mag
                
            new_row_a.append(final_angle)
            
        new_pos.append(new_row_a)
        
    t2a["xPositions"] = new_pos
    with open(t2a_path, "wb") as f:
        f.write(msgpack.packb(t2a, use_bin_type=True))
    print(f"  -> Migrated {path.name}")

if __name__ == "__main__":
    data_dir = Path("track/data")
    for d in data_dir.iterdir():
        if d.is_dir() and (d / "components" / "a").exists():
            migrate_dataset(d)
