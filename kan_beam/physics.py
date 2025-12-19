import numpy as np

PI2 = 2.0 * np.pi


def antenna_grid_positions(nx: int, ny: int, dx: float, dy: float, origin=(0.0, 0.0, 0.0)):
    """Create a rectangular aperture grid of antenna element positions.

    Args:
        nx: number of elements along x
        ny: number of elements along y
        dx: spacing along x (meters)
        dy: spacing along y (meters)
        origin: (x0, y0, z0) of the aperture center

    Returns:
        positions: (N, 3) array of (x, y, z) coordinates
        shape: (nx, ny)
    """
    x0, y0, z0 = origin
    xs = (np.arange(nx) - (nx - 1) / 2.0) * dx + x0
    ys = (np.arange(ny) - (ny - 1) / 2.0) * dy + y0
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    zv = np.full_like(xv, z0, dtype=float)
    positions = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
    return positions, (nx, ny)


def distances(ant_pos: np.ndarray, user_pos: np.ndarray):
    """Compute distances from each antenna to the user position.

    ant_pos: (N, 3)
    user_pos: (3,) or (B, 3)

    Returns:
        r: (N,) or (B, N) distances
    """
    ant_pos = np.asarray(ant_pos)
    user_pos = np.asarray(user_pos)
    if user_pos.ndim == 1:
        delta = ant_pos - user_pos[None, :]
        r = np.linalg.norm(delta, axis=-1)
        return r
    else:
        # batch
        delta = ant_pos[None, :, :] - user_pos[:, None, :]
        r = np.linalg.norm(delta, axis=-1)
        return r  # (B, N)


def near_field_channel(ant_pos: np.ndarray, user_pos: np.ndarray, wavelength: float):
    """Compute complex near-field channel for a single user.

    h_n = (1/r_n) * exp(-j 2*pi * r_n / lambda)

    Returns:
        h: (N,) complex64 vector
        r: (N,) distances
    """
    r = distances(ant_pos, user_pos)
    phase = -PI2 * r / wavelength
    amp = 1.0 / np.maximum(r, 1e-6)
    h = amp * np.exp(1j * phase)
    return h.astype(np.complex64), r


def beampattern(ant_pos: np.ndarray, weights: np.ndarray, grid_points: np.ndarray, wavelength: float):
    """Compute beampattern (complex field) at given grid points from weighted aperture.

    E(p) = sum_n w_n * exp(-j 2*pi * r_{n,p} / lambda) / r_{n,p}

    Args:
        ant_pos: (N, 3)
        weights: (N,) complex weights (unit-modulus phases or general)
        grid_points: (M, 3)
        wavelength: scalar
    Returns:
        field: (M,) complex values of field
    """
    ant_pos = np.asarray(ant_pos)
    weights = np.asarray(weights)
    gp = np.asarray(grid_points)
    delta = ant_pos[None, :, :] - gp[:, None, :]
    r = np.linalg.norm(delta, axis=-1)  # (M, N)
    phase = -PI2 * r / wavelength
    contrib = np.exp(1j * phase) / np.maximum(r, 1e-6)
    field = (contrib * weights[None, :]).sum(axis=1)
    return field


def make_focus_grid(center: np.ndarray, span_xy: float, nz: int, nx: int, ny: int):
    """Build a grid around a center point for evaluating focus sharpness.

    Args:
        center: (3,) focus point
        span_xy: +/- meters around center in x/y
        nz: number of z samples (useful for axial profile)
        nx, ny: grid resolution in x and y
    Returns:
        grid_points: (M, 3)
        shape: (nx, ny, nz)
    """
    cx, cy, cz = center
    xs = np.linspace(cx - span_xy, cx + span_xy, nx)
    ys = np.linspace(cy - span_xy, cy + span_xy, ny)
    zs = np.linspace(cz, cz, nz)
    xv, yv, zv = np.meshgrid(xs, ys, zs, indexing="xy")
    pts = np.stack([xv, yv, zv], axis=-1).reshape(-1, 3)
    return pts, (nx, ny, nz)
