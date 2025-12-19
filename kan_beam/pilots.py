import numpy as np


def generate_pilots(L: int, power: float = 1.0, seed: int = None):
    """Generate complex pilot symbols of length L.
    Returns (L,) complex numpy array with average power 'power'."""
    rng = np.random.default_rng(seed)
    # QPSK pilots for stability
    bits = rng.integers(0, 4, size=L)
    const = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex64)
    pilots = const[bits]
    pilots = pilots / np.sqrt(2)  # unit power
    pilots = pilots * np.sqrt(power)
    return pilots.astype(np.complex64)


def observe_pilots(h: np.ndarray, pilots: np.ndarray, noise_var: float = 0.0, seed: int = None):
    """Observation model for uplink pilots on each antenna:
    y_n[l] = pilot[l] * h_n + w_nl

    Args:
        h: (N,) complex channel vector
        pilots: (L,) complex pilot symbols
        noise_var: AWGN variance per complex sample
    Returns:
        Y: (N, L) complex observations per antenna per pilot
    """
    rng = np.random.default_rng(seed)
    N = h.shape[0]
    L = pilots.shape[0]
    Y = h[:, None] * pilots[None, :]
    if noise_var > 0.0:
        noise = (rng.normal(0, np.sqrt(noise_var/2), size=(N, L))
                 + 1j * rng.normal(0, np.sqrt(noise_var/2), size=(N, L))).astype(np.complex64)
        Y = Y + noise
    return Y.astype(np.complex64)


def to_real_imag_features(Y: np.ndarray):
    """Flatten complex observations to real-imag feature vector.
    Args:
        Y: (N, L) complex
    Returns:
        feats: (2*N*L,) float32
    """
    Y = np.asarray(Y)
    feats = np.concatenate([Y.real, Y.imag], axis=0).reshape(-1).astype(np.float32)
    return feats
