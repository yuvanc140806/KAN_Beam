import numpy as np
import torch


def unit_phase_weights_from_channel(h_real: torch.Tensor, h_imag: torch.Tensor):
    """Compute unit-modulus weights that conjugate the channel phases."""
    phase = torch.atan2(h_imag, h_real)
    return torch.exp(-1j * phase.detach().cpu().numpy())


def focus_gain(ant_pos: np.ndarray, weights: np.ndarray, target_point: np.ndarray, wavelength: float):
    from .physics import distances
    r = distances(ant_pos, target_point)
    phase = -2.0 * np.pi * r / wavelength
    field = (np.exp(1j * phase) / np.maximum(r, 1e-6) * weights).sum()
    return np.abs(field)
