import sys
import re

with open("Nrm/Play/PlayInt4/int1.py", "r") as f:
    code = f.read()

code = code.replace("from play6 import collect_segments\n", "")

inject = """
from dataclasses import dataclass
from analysis.tools.signal import preprocess_signal, hann_window_periodic
from analysis.tools.spectral import next_power_of_two
from analysis.Nrm.Tools.post_hit_regions import extract_post_hit_regions, EnabledRegionConfig

@dataclass(frozen=True)
class SegmentRecord:
    bond_id: int
    start_s: float
    stop_s: float
    mid_s: float
    spectrum: np.ndarray

def collect_segments(
    dataset: str,
    component: str,
    bond_ids: list[int],
    segment_len_s: float,
    overlap: float,
    *,
    bond_spacing_mode: str = "default",
) -> tuple[np.ndarray, list[SegmentRecord], np.ndarray]:
    records: list[SegmentRecord] = []
    nperseg: int | None = None
    freqs: np.ndarray | None = None

    for bond_id in bond_ids:
        result = extract_post_hit_regions(
            dataset=dataset,
            component=component,
            bond_id=bond_id,
            config=EnabledRegionConfig(bond_spacing_mode=str(bond_spacing_mode)),
        )
        processed, err = preprocess_signal(result.frame_times_s, result.signal, longest=False, handlenan=False)
        if processed is None:
            raise ValueError(f"Failed preprocessing bond {bond_id}: {err}")

        local_nperseg = max(8, int(round(segment_len_s * processed.Fs)))
        if local_nperseg > processed.y.size:
            continue
        step = max(1, local_nperseg - int(round(overlap * local_nperseg)))
        local_nfft = max(local_nperseg, next_power_of_two(local_nperseg))
        window = hann_window_periodic(local_nperseg)
        window_norm = float(np.sum(window))

        if nperseg is None:
            nperseg = local_nperseg
            freqs = np.fft.rfftfreq(local_nfft, d=processed.dt)
        elif local_nperseg != nperseg:
            raise ValueError("Segment length resolved differently across bonds")

        for start in range(0, processed.y.size - local_nperseg + 1, step):
            stop = start + local_nperseg
            segment = processed.y[start:stop] * window
            spectrum = np.fft.rfft(segment, n=local_nfft) / window_norm
            if spectrum.size > 2:
                spectrum = spectrum.copy()
                spectrum[1:-1] *= 2.0
            records.append(
                SegmentRecord(
                    bond_id=bond_id,
                    start_s=float(processed.t[start]),
                    stop_s=float(processed.t[stop - 1]),
                    mid_s=float(np.mean(processed.t[start:stop])),
                    spectrum=spectrum,
                )
            )

    if not records or freqs is None:
        raise ValueError("No valid segments could be collected from the specified bonds.")
    
    mean_amplitude = np.mean(np.abs([r.spectrum for r in records]), axis=0)
    return freqs, records, mean_amplitude

"""

code = code.replace("@dataclass(frozen=True)\nclass ResolvedPeak:", inject + "\n@dataclass(frozen=True)\nclass ResolvedPeak:")

with open("Nrm/Play/PlayInt4/int1.py", "w") as f:
    f.write(code)

