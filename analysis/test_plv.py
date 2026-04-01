import sys
import numpy as np

sys.path.insert(0, '/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis')
sys.path.insert(0, '/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/Nrm/Play')

from Nrm.Play.PlayInt4.int1 import collect_segments
from Nrm.Play.play7 import build_windows

def plv_for_bins(X, i1, i2, i3):
    X1 = X[:, i1]
    X2 = X[:, i2]
    X3 = X[:, i3]
    phase1 = np.angle(X1)
    phase2 = np.angle(X2)
    phase3 = np.angle(X3)
    phase_diff = phase1 + phase2 - phase3
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    return plv

def test_plv(f1, f2, f3):
    bond_ids = list(range(9))
    freqs_seg, records, mean_amplitude = collect_segments("CDX_10IC", "x", bond_ids, 100.0, 0.5, bond_spacing_mode="purecomoving")
    X = np.vstack([record.spectrum for record in records])
    
    i1 = np.argmin(np.abs(freqs_seg - f1))
    i2 = np.argmin(np.abs(freqs_seg - f2))
    i3 = np.argmin(np.abs(freqs_seg - f3))
    
    windows = build_windows(np.array([r.mid_s for r in records]), analysis_window_s=100.0, analysis_step_s=25.0, min_segments=6)
    
    max_plv = 0
    for window in windows:
        plv = plv_for_bins(X[window.segment_indices], i1, i2, i3)
        max_plv = max(max_plv, plv)
    
    # Calculate null distribution by phase randomization
    rng = np.random.default_rng(0)
    null_max_plvs = []
    for _ in range(60):
        max_null_plv = 0
        # For a pure phase null, we randomize the phases of the entire dataset first
        random_phase = rng.uniform(0, 2*np.pi, size=X.shape[0])
        for window in windows:
            idx = window.segment_indices
            phase1 = np.angle(X[idx, i1])
            phase2 = np.angle(X[idx, i2])
            phase3 = np.angle(X[idx, i3]) + random_phase[idx]  # shift one phase randomly
            plv = np.abs(np.mean(np.exp(1j * (phase1 + phase2 - phase3))))
            max_null_plv = max(max_null_plv, plv)
        null_max_plvs.append(max_null_plv)
        
    p_val = np.mean(np.array(null_max_plvs) >= max_plv)
    
    print(f"Triad {freqs_seg[i1]:.2f} + {freqs_seg[i2]:.2f} -> {freqs_seg[i3]:.2f} | Max PLV = {max_plv:.3f}, p-value = {p_val:.3f}")

print("--- PLV Negative Controls (Targeting 16.6 Fundamental) ---")
test_plv(8.96, 16.61 - 8.96, 16.61)
test_plv(12.0, 16.61 - 12.0, 16.61)
test_plv(3.35, 16.61 - 3.35, 16.61)

print("\n--- PLV Positive Controls (Real Cascades) ---")
test_plv(8.96, 9.42, 18.35)
