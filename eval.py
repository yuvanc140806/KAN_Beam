import numpy as np
import torch

from kan_beam.physics import antenna_grid_positions, near_field_channel, beampattern, make_focus_grid
from kan_beam.kan import ChannelEstimator
from kan_beam.pilots import generate_pilots, observe_pilots, to_real_imag_features
from kan_beam.beamforming import unit_phase_weights_from_channel


def evaluate_generalization(net: ChannelEstimator, wavelength: float, nx: int, ny: int, dx: float, dy: float,
                            L: int, noise_var: float = 1e-3, seed: int = 999):
    ant_pos, shape = antenna_grid_positions(nx, ny, dx, dy)
    rng = np.random.default_rng(seed)
    # unseen distances
    users = []
    for _ in range(64):
        x = rng.uniform(-nx * dx / 2, nx * dx / 2)
        y = rng.uniform(-ny * dy / 2, ny * dy / 2)
        z = rng.uniform(1.5, 4.0)
        users.append(np.array([x, y, z], dtype=np.float32))
    users = np.stack(users, axis=0)
    pilots = generate_pilots(L, power=1.0, seed=1234)

    nmse_list = []
    for user in users:
        h_true, _ = near_field_channel(ant_pos, user, wavelength)
        Y = observe_pilots(h_true, pilots, noise_var=noise_var)
        feats = to_real_imag_features(Y)
        with torch.no_grad():
            pred = net(torch.from_numpy(feats[None, :]).float())
        N = nx * ny
        pr = pred[0, :N].numpy()
        pi = pred[0, N:].numpy()
        h_pred = pr + 1j * pi
        nmse = np.mean(np.abs(h_pred - h_true) ** 2) / np.mean(np.abs(h_true) ** 2)
        nmse_list.append(nmse)
    return float(np.mean(nmse_list))


def evaluate_beam_focus(net: ChannelEstimator, wavelength: float, nx: int, ny: int, dx: float, dy: float,
                         user: np.ndarray):
    ant_pos, shape = antenna_grid_positions(nx, ny, dx, dy)
    h_true, _ = near_field_channel(ant_pos, user, wavelength)
    pilots = generate_pilots(4, power=1.0, seed=777)
    Y = observe_pilots(h_true, pilots, noise_var=1e-3)
    feats = to_real_imag_features(Y)
    with torch.no_grad():
        pred = net(torch.from_numpy(feats[None, :]).float())
    N = nx * ny
    pr = pred[0, :N].numpy()
    pi = pred[0, N:].numpy()
    weights = unit_phase_weights_from_channel(torch.from_numpy(pr), torch.from_numpy(pi))

    # Compute field around target
    grid, shape3 = make_focus_grid(center=user, span_xy=0.5, nz=1, nx=51, ny=51)
    field = beampattern(ant_pos, weights, grid, wavelength)
    power = np.abs(field) ** 2
    peak = power.max()
    return float(peak), power.reshape(shape3)


if __name__ == "__main__":
    # Load a trained model via simple training
    from train import Config, train
    cfg = Config()
    net, ds = train(cfg)
    gen_nmse = evaluate_generalization(net, cfg.wavelength, cfg.nx, cfg.ny, cfg.dx, cfg.dy, cfg.L, cfg.noise_var)
    print(f"Generalization NMSE: {gen_nmse:.4f}")
    user = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    peak, power = evaluate_beam_focus(net, cfg.wavelength, cfg.nx, cfg.ny, cfg.dx, cfg.dy, user)
    print(f"Beam focus peak power: {peak:.4f}")
