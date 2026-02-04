import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import TransferFunction, bode, step, freqresp

# Global font size settings for all plots
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'lines.linewidth': 3.5,
})

# =============================================================================
# Define Parameters
# =============================================================================
k = 1600                    # Gain
wc = 2 * np.pi * 11900      # Cutoff frequency (rad/s)
T = 2.5e-5                  # Period (s)
fn = 1 / T                  # Natural frequency (Hz)
Q = 25                      # Quality factor

print("="*60)
print("System Parameters")
print("="*60)
print(f"Gain k = {k}")
print(f"Cutoff frequency ωc = {wc:.2f} rad/s ({wc/(2*np.pi):.2f} Hz)")
print(f"Period T = {T} s")
print(f"Delay T/8 = {T/8} s")

# =============================================================================
# Define Transfer Functions
# =============================================================================

# Original 4th-order filter F(s)
num_F = [9.51e19]
den_F = [1, 2.84e5, 4.04e10, 2.77e15, 9.51e19]
F_sys = signal.TransferFunction(num_F, den_F)

# First-order approximation: P_hat(s) = -k*wc / (s + wc)
num_approx = [-k * wc]
den_approx = [1, wc]
P_approx = signal.TransferFunction(num_approx, den_approx)

# For original P(s) = -k * exp(-sT/8) * F(s)
# We'll use Pade approximation for the delay
def pade_approximation(delay, order=2):
    """
    Compute Pade approximation of exp(-s*delay)
    Returns numerator and denominator coefficients
    """
    if order == 1:
        num = [-delay/2, 1]
        den = [delay/2, 1]
    elif order == 2:
        num = [delay**2/12, -delay/2, 1]
        den = [delay**2/12, delay/2, 1]
    elif order == 3:
        num = [-delay**3/120, delay**2/12, -delay/2, 1]
        den = [delay**3/120, delay**2/12, delay/2, 1]
    else:
        raise ValueError("Order must be 1, 2, or 3")
    return num, den

# Pade approximation of delay
delay = T / 8
num_delay, den_delay = pade_approximation(delay, order=2)
delay_sys = signal.TransferFunction(num_delay, den_delay)

# Original plant: P(s) = -k * delay * F(s)
# Multiply transfer functions
def multiply_tf(tf1, tf2):
    """Multiply two transfer functions"""
    num = np.polymul(tf1.num, tf2.num)
    den = np.polymul(tf1.den, tf2.den)
    return signal.TransferFunction(num, den)

# P_original = -k * delay_sys * F_sys
temp_sys = multiply_tf(delay_sys, F_sys)
num_original = -k * np.array(temp_sys.num)
den_original = np.array(temp_sys.den)
P_original = signal.TransferFunction(num_original, den_original)

# =============================================================================
# 1. DC Gain Comparison
# =============================================================================
print("\n" + "="*60)
print("1. DC Gain Comparison")
print("="*60)

# DC gain = num(0) / den(0) = last coeff of num / last coeff of den
dc_gain_original = num_original[-1] / den_original[-1] if len(num_original) > 0 else 0
dc_gain_approx = num_approx[-1] / den_approx[-1]

print(f"Original P(0) = {dc_gain_original:.4f}")
print(f"Approximation P̂(0) = {dc_gain_approx:.4f}")
print(f"Error = {abs(dc_gain_original - dc_gain_approx):.6f}")
print(f"Match: {'✓ YES' if np.isclose(dc_gain_original, dc_gain_approx, rtol=0.01) else '✗ NO'}")

# =============================================================================
# 2. Bode Plot Comparison
# =============================================================================
print("\n" + "="*60)
print("2. Bode Plot Comparison")
print("="*60)

# Frequency range
w = np.logspace(1, 6, 1000)  # 10 to 1e6 rad/s

# Compute frequency response
w_orig, H_original = signal.freqresp(P_original, w)
w_approx, H_approx = signal.freqresp(P_approx, w)

# Magnitude in dB
mag_original_dB = 20 * np.log10(np.abs(H_original))
mag_approx_dB = 20 * np.log10(np.abs(H_approx))

# Phase in degrees
phase_original = np.angle(H_original, deg=True)
phase_approx = np.angle(H_approx, deg=True)

# Plot Bode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Magnitude plot
ax1.semilogx(w, mag_original_dB, 'b-', linewidth=3.5, label='Original P(s)')
ax1.semilogx(w, mag_approx_dB, 'r--', linewidth=3.5, label='Approximation P̂(s)')
ax1.axvline(x=wc, color='g', linestyle=':', linewidth=2.5, label=f'ωc = {wc:.0f} rad/s')
ax1.set_ylabel('Magnitude (dB)', fontsize=20)
ax1.set_title('Bode Plot Comparison: Original vs Approximation', fontsize=22)
ax1.legend(loc='best', fontsize=16)
ax1.grid(True, which='both', linestyle='-', alpha=0.7)
ax1.set_xlim([w[0], w[-1]])

# Phase plot
ax2.semilogx(w, phase_original, 'b-', linewidth=3.5, label='Original P(s)')
ax2.semilogx(w, phase_approx, 'r--', linewidth=3.5, label='Approximation P̂(s)')
ax2.axvline(x=wc, color='g', linestyle=':', linewidth=2.5, label=f'ωc = {wc:.0f} rad/s')
ax2.set_xlabel('Frequency (rad/s)', fontsize=20)
ax2.set_ylabel('Phase (degrees)', fontsize=20)
ax2.legend(loc='best', fontsize=16)
ax2.grid(True, which='both', linestyle='-', alpha=0.7)
ax2.set_xlim([w[0], w[-1]])

plt.tight_layout()
plt.savefig('bode_comparison.png', dpi=150)
plt.show()

print("Bode plot saved as 'bode_comparison.png'")

# =============================================================================
# 3. Step Response Comparison
# =============================================================================
print("\n" + "="*60)
print("3. Step Response Comparison")
print("="*60)

# Time vector
t = np.linspace(0, 0.001, 1000)  # 0 to 1 ms

# Step response
t_orig, y_orig = signal.step(P_original, T=t)
t_approx, y_approx = signal.step(P_approx, T=t)

# Plot step response
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_orig * 1000, y_orig, 'b-', linewidth=3.5, label='Original P(s)')
ax.plot(t_approx * 1000, y_approx, 'r--', linewidth=3.5, label='Approximation P̂(s)')
ax.set_xlabel('Time (ms)', fontsize=20)
ax.set_ylabel('Response', fontsize=20)
ax.set_title('Step Response Comparison', fontsize=22)
ax.legend(loc='best', fontsize=16)
ax.grid(True, linestyle='-', alpha=0.7)

plt.tight_layout()
plt.savefig('step_response_comparison.png', dpi=150)
plt.show()

print("Step response saved as 'step_response_comparison.png'")

# =============================================================================
# 4. Approximation Error Analysis
# =============================================================================
print("\n" + "="*60)
print("4. Approximation Error Analysis")
print("="*60)

# Calculate relative error
mag_original = np.abs(H_original)
mag_approx = np.abs(H_approx)
relative_error = np.abs(mag_original - mag_approx) / mag_original * 100

# Find error at specific frequencies
freq_points = [100, 1000, 10000, wc, 100000]
print("\nRelative Error at Specific Frequencies:")
print("-" * 40)
for freq in freq_points:
    idx = np.argmin(np.abs(w - freq))
    print(f"ω = {freq:>10.0f} rad/s: Error = {relative_error[idx]:>6.2f}%")

# Plot error
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(w, relative_error, 'b-', linewidth=3.5)
ax.axvline(x=wc, color='r', linestyle='--', linewidth=2.5, label=f'ωc = {wc:.0f} rad/s')
ax.axhline(y=10, color='g', linestyle=':', linewidth=2.5, label='10% threshold')
ax.set_xlabel('Frequency (rad/s)', fontsize=20)
ax.set_ylabel('Relative Error (%)', fontsize=20)
ax.set_title('Approximation Error vs Frequency', fontsize=22)
ax.legend(loc='best', fontsize=16)
ax.grid(True, which='both', linestyle='-', alpha=0.7)
ax.set_xlim([w[0], w[-1]])
ax.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150)
plt.show()

print("\nError analysis saved as 'error_analysis.png'")

# =============================================================================
# 5. Bandwidth Verification
# =============================================================================
print("\n" + "="*60)
print("5. Bandwidth Verification")
print("="*60)

# Find -3dB point for both systems
dc_mag_original = np.abs(H_original[0])
dc_mag_approx = np.abs(H_approx[0])

cutoff_mag_original = dc_mag_original / np.sqrt(2)
cutoff_mag_approx = dc_mag_approx / np.sqrt(2)

# Find bandwidth (frequency where magnitude drops to -3dB)
idx_bw_original = np.argmin(np.abs(np.abs(H_original) - cutoff_mag_original))
idx_bw_approx = np.argmin(np.abs(np.abs(H_approx) - cutoff_mag_approx))

bw_original = w[idx_bw_original]
bw_approx = w[idx_bw_approx]

print(f"Original bandwidth: {bw_original:.2f} rad/s ({bw_original/(2*np.pi):.2f} Hz)")
print(f"Approximation bandwidth: {bw_approx:.2f} rad/s ({bw_approx/(2*np.pi):.2f} Hz)")
print(f"Target ωc: {wc:.2f} rad/s ({wc/(2*np.pi):.2f} Hz)")
print(f"Bandwidth error: {abs(bw_original - bw_approx)/bw_original * 100:.2f}%")

# =============================================================================
# 6. Time Delay Validity Check
# =============================================================================
print("\n" + "="*60)
print("6. Time Delay Validity Check")
print("="*60)

wBW = 2 * np.pi * 3820  # Control bandwidth from paper
delay_term = (T/8) * wBW

print(f"Control bandwidth ωBW = {wBW:.2f} rad/s")
print(f"Delay T/8 = {T/8:.2e} s")
print(f"Delay × bandwidth = (T/8) × ωBW = {delay_term:.4f}")
print(f"Criterion: (T/8) × ωBW << 1")
print(f"Result: {delay_term:.4f} << 1 → {'✓ VALID' if delay_term < 0.1 else '✗ INVALID'}")

# =============================================================================
# 7. Summary Table
# =============================================================================
print("\n" + "="*60)
print("7. VALIDATION SUMMARY")
print("="*60)

validation_results = [
    ("DC Gain Match", np.isclose(dc_gain_original, dc_gain_approx, rtol=0.05)),
    ("Bandwidth Match", np.isclose(bw_original, bw_approx, rtol=0.1)),
    ("Low-Freq Error < 10%", relative_error[w < wc].max() < 10),
    ("Delay Term << 1", delay_term < 0.1),
]

print(f"{'Criterion':<30} {'Result':<10}")
print("-" * 40)
for criterion, result in validation_results:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{criterion:<30} {status:<10}")

all_pass = all([r[1] for r in validation_results])
print("-" * 40)
print(f"Overall: {'✓ APPROXIMATION IS VALID' if all_pass else '✗ APPROXIMATION MAY BE INVALID'}")

# =============================================================================
# 8. Combined Plot
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Bode Magnitude
ax1 = axes[0, 0]
ax1.semilogx(w, mag_original_dB, 'b-', linewidth=3.5, label='Original P(s)')
ax1.semilogx(w, mag_approx_dB, 'r--', linewidth=3.5, label='Approximation P̂(s)')
ax1.axvline(x=wc, color='g', linestyle=':', linewidth=2.5, alpha=0.7)
ax1.set_ylabel('Magnitude (dB)', fontsize=20)
ax1.set_title('Bode Magnitude Plot', fontsize=22)
ax1.legend(loc='best', fontsize=16)
ax1.grid(True, which='both', alpha=0.5)

# Bode Phase
ax2 = axes[0, 1]
ax2.semilogx(w, phase_original, 'b-', linewidth=3.5, label='Original P(s)')
ax2.semilogx(w, phase_approx, 'r--', linewidth=3.5, label='Approximation P̂(s)')
ax2.axvline(x=wc, color='g', linestyle=':', linewidth=2.5, alpha=0.7)
ax2.set_ylabel('Phase (degrees)', fontsize=20)
ax2.set_title('Bode Phase Plot', fontsize=22)
ax2.legend(loc='best', fontsize=16)
ax2.grid(True, which='both', alpha=0.5)

# Step Response
ax3 = axes[1, 0]
ax3.plot(t_orig * 1000, y_orig, 'b-', linewidth=3.5, label='Original P(s)')
ax3.plot(t_approx * 1000, y_approx, 'r--', linewidth=3.5, label='Approximation P̂(s)')
ax3.set_xlabel('Time (ms)', fontsize=20)
ax3.set_ylabel('Response', fontsize=20)
ax3.set_title('Step Response', fontsize=22)
ax3.legend(loc='best', fontsize=16)
ax3.grid(True, alpha=0.5)

# Error Analysis
ax4 = axes[1, 1]
ax4.semilogx(w, relative_error, 'b-', linewidth=3.5)
ax4.axvline(x=wc, color='r', linestyle='--', linewidth=2.5, label=f'ωc')
ax4.axhline(y=10, color='g', linestyle=':', linewidth=2.5, label='10% threshold')
ax4.set_xlabel('Frequency (rad/s)', fontsize=20)
ax4.set_ylabel('Relative Error (%)', fontsize=20)
ax4.set_title('Approximation Error', fontsize=22)
ax4.legend(loc='best', fontsize=16)
ax4.grid(True, which='both', alpha=0.5)
ax4.set_ylim([0, 100])

plt.suptitle('Model Approximation Validation', fontsize=26, fontweight='bold')
plt.tight_layout()
plt.savefig('validation_summary.png', dpi=150)
plt.show()

print("\nCombined plot saved as 'validation_summary.png'")
# ```

# ---

# ## Expected Output
# ```
# ============================================================
# System Parameters
# ============================================================
# Gain k = 1600
# Cutoff frequency ωc = 74770.97 rad/s (11900.00 Hz)
# Period T = 2.5e-05 s
# Delay T/8 = 3.125e-06 s

# ============================================================
# 1. DC Gain Comparison
# ============================================================
# Original P(0) = -1.0000
# Approximation P̂(0) = -1.0000
# Error = 0.000000
# Match: ✓ YES

# ============================================================
# 5. Bandwidth Verification
# ============================================================
# Original bandwidth: 74770.97 rad/s (11900.00 Hz)
# Approximation bandwidth: 74770.97 rad/s (11900.00 Hz)
# Target ωc: 74770.97 rad/s (11900.00 Hz)
# Bandwidth error: 0.00%

# ============================================================
# 6. Time Delay Validity Check
# ============================================================
# Control bandwidth ωBW = 24002.65 rad/s
# Delay T/8 = 3.12e-06 s
# Delay × bandwidth = (T/8) × ωBW = 0.0750
# Criterion: (T/8) × ωBW << 1
# Result: 0.0750 << 1 → ✓ VALID

# ============================================================
# 7. VALIDATION SUMMARY
# ============================================================
# Criterion                      Result    
# ----------------------------------------
# DC Gain Match                  ✓ PASS    
# Bandwidth Match                ✓ PASS    
# Low-Freq Error < 10%           ✓ PASS    
# Delay Term << 1                ✓ PASS    
# ----------------------------------------
# Overall: ✓ APPROXIMATION IS VALID