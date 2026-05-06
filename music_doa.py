
"""
MUSIC Algorithm for Maritime Radar DOA Estimation
=================================================
Key Concepts:
- Eigen decomposition separates signal and noise subspaces
- MUSIC exploits orthogonality for super resolution DOA
- Conventional beamforming is limited by Rayleigh resolution
- MUSIC breaks this limit via subspace geometry

Author: Piyush Kapoor 
Context: Naval radar, multiple target tracking, electronic warfare
"""

import numpy as np
from numpy.linalg import eigh

# ---------------------------------------------------------
# 1. ARRAY & SCENARIO PARAMETERS
# ---------------------------------------------------------
M = 16                      # Array elements (ULA)
d = 0.5                     # Element spacing (wavelengths)
N = 512                     # Snapshots for covariance estimate

#maritime targets: fishing vessel, aircraft, drone swarm, container ship
# Maritime targets: [angle_deg, range_km, rcs_m2, speed_ms]
TARGETS = {
    'fast_attack_craft':  {'theta': 12,  'range': 5,   'rcs': 2.0,  'v': 25},
    'fishing_vessel':     {'theta': 35,  'range': 12,  'rcs': 15.0, 'v': 8},
    'container_ship':     {'theta': 68,  'range': 25,  'rcs': 1000, 'v': 15},
    'swarm_drone':        {'theta': -20, 'range': 3,   'rcs': 0.5,  'v': 35},
}

# Environment
SEA_STATE = 5               # Rough seas (2.5-4m waves)
WIND_KTS = 25               # 25 knotts
NOISE_DBM = -25             # Thermal noise floor

# ---------------------------------------------------------
# 2. STEERING VECTOR (ULA phase progression)
# ---------------------------------------------------------
def steering_vector(theta, M, d):
    """Generate array manifold vector for direction theta (degrees)"""
    theta_rad = np.deg2rad(theta)
    n = np.arange(M)
    return np.exp(-1j * 2 * np.pi * d * n * np.sin(theta_rad)) 

# ---------------------------------------------------------
# 3. SIGNAL GENERATION (Maritime scenario)
# ---------------------------------------------------------
def generate_maritime_signals(M, N, targets, sea_state, noise_dbm):
    """
    Generate real maritime radar returns:
    - Target signals with RCS dependent amplitude
    - Spatially correlated sea clutter 
    - Spatial thermal noise
    """
    X = np.zeros((M, N), dtype=complex)

    # --- Targets ---
    for t in range(N):
        snapshot = np.zeros(M, dtype=complex)
        for name, p in targets.items():
            # Radar equation: power ~ RCS / range^4
            amp = np.sqrt(p['rcs'] / p['range']**4)

            # Doppler phase (simplified)
            fd = 2 * p['v'] * 9.5e9 / 3e8  # X-band
            doppler = np.exp(1j * 2 * np.pi * fd * t / 2000)

            # Swerling fluctuation
            fluct = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)

            a = steering_vector(p['theta'], M, 0.5)
            snapshot += amp * fluct * doppler * a
        X[:, t] = snapshot

    # --- Sea Clutter (distributed, spatially correlated) ---
    clutter_angles = np.linspace(-20, 20, 40)  # Sea surface sector
    X_clutter = np.zeros((M, N), dtype=complex)
    for t in range(N):
        for ca in clutter_angles:
            c_amp = 0.03 * (np.random.randn() + 1j*np.random.randn())
            X_clutter[:, t] += c_amp * steering_vector(ca, M, 0.5)

    # --- Thermal Noise (spatially white) ---
    noise_power = 10**(noise_dbm/10)
    X_noise = np.sqrt(noise_power/2) * (
        np.random.randn(M, N) + 1j*np.random.randn(M, N)
    )

    return X + X_clutter + X_noise

# ---------------------------------------------------------
# 4. MUSIC ALGORITHM
# ---------------------------------------------------------
def music_doa(R, M, K_est, d, angles=np.linspace(-90, 90, 1801)):
    """
    Multiple Signal Classification (MUSIC) DOA estimation

    Parameters:
        R: Sample covariance matrix (MxM)
        M: Number of array elements
        K_est: Estimated number of signals
        d: Element spacing (wavelengths)
        angles: Scan grid (degrees)

    Returns:
        P_music: Pseudo-spectrum (dB, normalized)
        detected_angles: Peak locations
    """
    # 4.1 Eigen-decomposition
    eigenvalues, eigenvectors = eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # 4.2 Subspace separation
    E_signal = eigenvectors[:, :K_est]      # Signal subspace
    E_noise = eigenvectors[:, K_est:]       # Noise subspace (M-K dimensions)

    # 4.3 Pseudo-spectrum: scan for orthogonality
    P_music = np.zeros(len(angles))

    for i, theta in enumerate(angles):
        a = steering_vector(theta, M, d)
        # Projection of steering vector onto noise subspace
        projection = a.conj().T @ E_noise @ E_noise.conj().T @ a
        # Invert: orthogonal -> peak
        P_music[i] = 1.0 / (np.abs(projection) + 1e-12)

    # Normalize to dB
    P_music = 10 * np.log10(P_music / np.max(P_music))

    # Peak detection
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(P_music, height=-20, distance=50)
    detected_angles = angles[peaks]

    return P_music, detected_angles

# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # Generate data
    X = generate_maritime_signals(M, N, TARGETS, SEA_STATE, NOISE_DBM)

    # Covariance matrix
    R = (X @ X.conj().T) / N

    # Estimate K (number of sources) via eigenvalue threshold
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(eigvals)[::-1]
    noise_floor = np.mean(eigvals[M//2:])
    K_est = int(np.sum(eigvals > 3 * noise_floor))

    print(f"Array: {M}-element ULA | Sea State: {SEA_STATE}")
    print(f"Estimated signals: {K_est}")
    print(f"Eigenvalues: {np.round(eigvals[:6], 3)}")

    # Run MUSIC
    angles = np.linspace(-90, 90, 3601)
    P_music, detected = music_doa(R, M, K_est, 0.5, angles)

    print(f"\nDetected angles: {np.round(detected, 1)}°")
    print("\nTrue vs Estimated:")
    for name, p in TARGETS.items():
        closest = detected[np.argmin(np.abs(detected - p['theta']))]
        print(f"  {name:18s}: {p['theta']:3d}° → {closest:5.1f}°  (err={abs(closest-p['theta']):.1f}°)")

    # Plotting (matplotlib)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(angles, P_music, 'b-', linewidth=1.5, label='MUSIC Pseudo-Spectrum')

    colors = ['red', 'green', 'orange', 'purple']
    for idx, (name, p) in enumerate(TARGETS.items()):
        ax.axvline(x=p['theta'], color=colors[idx], linestyle='--', 
                   alpha=0.6, label=f'{name} ({p["theta"]}°)')

    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Power (dB)')
    ax.set_title(f'MUSIC DOA: Maritime Radar | {M}-Element ULA | Sea State {SEA_STATE}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-90, 90])
    plt.tight_layout()
    plt.show()
