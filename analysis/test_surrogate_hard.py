import sys
import numpy as np

sys.path.insert(0, '/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis')
sys.path.insert(0, '/home/gram/Documents/FileFolder/Projects/BlocksSSH/analysis/Nrm/Play')

from Nrm.Play.PlayInt4.int1 import collect_segments
from Nrm.Play.play7 import build_windows, score_windows, score_windows_surrogate, empirical_pvalue

def test_surrogate(f1, f2, f3):
    bond_ids = list(range(9))
    freqs_seg, records, mean_amplitude = collect_segments("CDX_10IC", "x", bond_ids, 100.0, 0.5, bond_spacing_mode="comoving")
    X = np.vstack([record.spectrum for record in records])
    
    i1 = np.argmin(np.abs(freqs_seg - f1))
    i2 = np.argmin(np.abs(freqs_seg - f2))
    i3 = np.argmin(np.abs(freqs_seg - f3))
    
    windows = build_windows(np.array([r.mid_s for r in records]), analysis_window_s=100.0, analysis_step_s=25.0, min_segments=6)
    
    # Real score
    b2_scores = score_windows(X, windows, i1=i1, i2=i2, i3=i3)
    max_b2 = np.max(b2_scores)
    
    # Surrogate score
    rng = np.random.default_rng(0)
    null_scores = []
    for _ in range(60):
        s_scores = score_windows_surrogate(X, windows, i1=i1, i2=i2, i3=i3, rng=rng)
        null_scores.append(np.max(s_scores))
    null_scores = np.array(null_scores)
    p_val = empirical_pvalue(max_b2, null_scores)
    
    print(f"Triad {freqs_seg[i1]:.2f} + {freqs_seg[i2]:.2f} -> {freqs_seg[i3]:.2f} | Max b^2 = {max_b2:.3f}, p-value = {p_val:.3f}")

print("--- Surrogate Negative Controls (Targeting 16.6 Fundamental) ---")
test_surrogate(8.96, 16.61 - 8.96, 16.61)
test_surrogate(12.0, 16.61 - 12.0, 16.61)
test_surrogate(3.35, 16.61 - 3.35, 16.61)

print("\n--- Surrogate Positive Controls (Real Cascades) ---")
test_surrogate(8.96, 9.42, 18.35)
