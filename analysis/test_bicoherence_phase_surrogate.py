import sys
import numpy as np

sys.path.insert(0, '/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis')
sys.path.insert(0, '/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/Nrm/Play')

from Nrm.Play.PlayInt4.int1 import collect_segments
from Nrm.Play.play7 import build_windows

def bicoherence_for_bins(X, i1, i2, i3):
    X1 = X[:, i1]
    X2 = X[:, i2]
    X3 = X[:, i3]
    z = X1 * X2 * np.conj(X3)
    denom = np.mean(np.abs(X1 * X2) ** 2) * np.mean(np.abs(X3) ** 2)
    return 0.0 if denom <= 1e-18 else float(np.abs(np.mean(z)) ** 2 / denom)

def test_phase_surrogate(f1, f2, f3):
    bond_ids = list(range(9))
    freqs_seg, records, mean_amplitude = collect_segments("CDX_10IC", "x", bond_ids, 100.0, 0.5, bond_spacing_mode="comoving")
    X = np.vstack([record.spectrum for record in records])
    
    i1 = np.argmin(np.abs(freqs_seg - f1))
    i2 = np.argmin(np.abs(freqs_seg - f2))
    i3 = np.argmin(np.abs(freqs_seg - f3))
    
    windows = build_windows(np.array([r.mid_s for r in records]), analysis_window_s=100.0, analysis_step_s=25.0, min_segments=6)
    
    max_b2 = 0
    for window in windows:
        b2 = bicoherence_for_bins(X[window.segment_indices], i1, i2, i3)
        max_b2 = max(max_b2, b2)
    
    # Surrogate score (Phase randomization)
    rng = np.random.default_rng(0)
    null_max_b2 = []
    
    for _ in range(60):
        # Generate random phases for the whole dataset
        rand_phase = rng.uniform(0, 2*np.pi, size=X.shape[0])
        max_null = 0
        for window in windows:
            idx = window.segment_indices
            X1 = X[idx, i1]
            X2 = X[idx, i2]
            X3 = X[idx, i3] * np.exp(1j * rand_phase[idx])
            
            z = X1 * X2 * np.conj(X3)
            denom = np.mean(np.abs(X1 * X2) ** 2) * np.mean(np.abs(X3) ** 2)
            null_b2 = 0.0 if denom <= 1e-18 else float(np.abs(np.mean(z)) ** 2 / denom)
            max_null = max(max_null, null_b2)
            
        null_max_b2.append(max_null)
        
    p_val = np.mean(np.array(null_max_b2) >= max_b2)
    
    print(f"Triad {freqs_seg[i1]:.2f} + {freqs_seg[i2]:.2f} -> {freqs_seg[i3]:.2f} | Max b^2 = {max_b2:.3f}, p-value = {p_val:.3f}")

print("--- Phase Surrogate Negative Controls (Targeting 16.6 Fundamental) ---")
test_phase_surrogate(8.96, 16.61 - 8.96, 16.61)
test_phase_surrogate(12.0, 16.61 - 12.0, 16.61)
test_phase_surrogate(3.35, 16.61 - 3.35, 16.61)

print("\n--- Phase Surrogate Positive Controls (Real Cascades) ---")
test_phase_surrogate(8.96, 9.42, 18.35)
test_phase_surrogate(0.41, 16.61, 17.02)
