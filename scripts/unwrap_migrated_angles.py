import msgpack
import numpy as np
from pathlib import Path

def unwrap_dataset(path: Path):
    t2a_path = path / "components" / "a" / "track2_permanence.msgpack"
    if not t2a_path.exists():
        return

    with open(t2a_path, "rb") as f:
        t2a = msgpack.unpackb(f.read(), raw=False)

    arr = np.array(t2a["xPositions"])
    if arr.size == 0:
        return
        
    jumps_before = 0
    for col in range(arr.shape[1]):
        col_data = arr[:, col]
        mask = np.isfinite(col_data)
        if np.any(mask):
            jumps_before += np.sum(np.abs(np.diff(col_data[mask])) > 1.0)
            arr[mask, col] = np.unwrap(col_data[mask], period=np.pi)

    if jumps_before > 0:
        print(f"Unwrapping {path.name}... (fixed {jumps_before} wrap jumps)")
        t2a["xPositions"] = arr.tolist()
        with open(t2a_path, "wb") as f:
            f.write(msgpack.packb(t2a, use_bin_type=True))

if __name__ == "__main__":
    data_dir = Path("track/data")
    for d in data_dir.iterdir():
        if d.is_dir() and (d / "components" / "a").exists():
            unwrap_dataset(d)
