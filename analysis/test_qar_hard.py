import sys
import numpy as np
import scipy.signal as sp_signal

sys.path.insert(0, '/home/gram/Documents/FileFolder/Projects/BlocksSSH')

from analysis.tools.bonds import load_bond_signal_dataset
from analysis.tools.signal import preprocess_signal
from analysis.tools.io import load_track2_dataset
from analysis.Nrm.Play.PlayInt5AR.int1_qar import extract_analytic_envelope, fit_qar

def test_triad(f1, f2, f3):
    bond_dataset = load_bond_signal_dataset(dataset='CDX_10IC_x', bond_spacing_mode='purecomoving', component='x')
    track2 = load_track2_dataset(dataset='CDX_10IC_x')
    t = track2.frame_times_s
    
    processed_signals = []
    processed_t = None
    for i in range(bond_dataset.signal_matrix.shape[1]):
        sig = bond_dataset.signal_matrix[:, i]
        processed, _ = preprocess_signal(t, sig)
        if processed:
            if processed_t is None: processed_t = processed.t
            processed_signals.append(processed.y)
    
    y_agg = np.mean(processed_signals, axis=0)
    fs = 1.0 / np.median(np.diff(processed_t))
    
    Z1 = extract_analytic_envelope(y_agg, fs, f1, 0.2)
    Z2 = extract_analytic_envelope(y_agg, fs, f2, 0.2)
    Z3 = extract_analytic_envelope(y_agg, fs, f3, 0.2)
    
    win_samples = int(15.0 * fs)
    step_samples = int(2.5 * fs)
    
    p_vals = []
    for start in range(0, len(y_agg) - win_samples, step_samples):
        _, p = fit_qar(Z1[start:start+win_samples], Z2[start:start+win_samples], Z3[start:start+win_samples], 2)
        p_vals.append(p)
    log_p = -np.log10(np.maximum(p_vals, 1e-15))
    print(f'Triad {f1:.2f} + {f2:.2f} -> {f3:.2f} | Max -log10(p) = {np.max(log_p):.2f}, % windows > 3.0 = {100*np.mean(log_p > 3.0):.1f}%')

# 16.61 is a KNOWN FUNDAMENTAL. We force mathematical sums to equal 16.61 exactly.
print("--- Negative Controls (Targeting 16.61 Fundamental) ---")
test_triad(8.96, 16.61 - 8.96, 16.61)
test_triad(12.0, 16.61 - 12.0, 16.61)
test_triad(3.35, 16.61 - 3.35, 16.61)

print("\n--- Positive Controls (Real Cascades) ---")
test_triad(8.96, 9.42, 18.35)  # 8.96 + 9.42 = 18.38 ~ 18.35 (within df=0.2)
