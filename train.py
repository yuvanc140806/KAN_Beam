import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from kan_beam.physics import antenna_grid_positions, near_field_channel
from kan_beam.pilots import generate_pilots, observe_pilots, to_real_imag_features
from kan_beam.kan import ChannelEstimator
from kan_beam.losses import physics_informed_loss


def nmse(pred: torch.Tensor, target: torch.Tensor):
    return torch.mean(torch.sum((pred - target) ** 2, dim=-1) / torch.sum(target ** 2, dim=-1)).item()


@dataclass
class Config:
    nx: int = 8
    ny: int = 8
    dx: float = 0.5  # meters
    dy: float = 0.5
    wavelength: float = 0.1  # 3 GHz approx (c ~ 3e8, f ~ 3e9) => lambda ~ 0.1 m
    L: int = 4  # pilots
    noise_var: float = 1e-3
    samples: int = 512
    val_samples: int = 128
    epochs: int = 30  # reduced for faster training
    batch_size: int = 64  # larger batch for GPU efficiency
    width: int = 256
    depth: int = 3
    K: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # auto-detect GPU
    patience: int = 5  # early stopping patience
    lr: float = 1e-3
    lr_schedule: str = "cosine"  # "cosine" or "constant"
    val_interval: int = 2  # validate every N epochs
    use_amp: bool = True  # mixed precision training
    use_compile: bool = False  # torch.compile optimization
    num_workers: int = 2  # data loading parallelism


class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg: Config, seed: int = 42):
        super().__init__()
        print(f"Pre-computing dataset (seed={seed})...")
        device = torch.device(cfg.device)
        
        rng = np.random.default_rng(seed)
        ant_pos, shape = antenna_grid_positions(cfg.nx, cfg.ny, cfg.dx, cfg.dy)
        self.ant_pos = ant_pos
        self.shape = shape
        self.wavelength = cfg.wavelength
        self.L = cfg.L
        self.noise_var = cfg.noise_var
        self.device = device
        
        # Generate user positions (CPU)
        users = []
        for _ in range(cfg.samples):
            x = rng.uniform(-cfg.nx * cfg.dx / 2, cfg.nx * cfg.dx / 2)
            y = rng.uniform(-cfg.ny * cfg.dy / 2, cfg.ny * cfg.dy / 2)
            z = rng.uniform(0.5, 3.0)
            users.append(np.array([x, y, z], dtype=np.float32))
        self.users = np.stack(users, axis=0)  # (S, 3)
        
        # GPU tensors for batch computation
        ant_pos_torch = torch.from_numpy(ant_pos).float().to(device)  # (N, 3)
        users_torch = torch.from_numpy(self.users).float().to(device)  # (S, 3)
        pilots = generate_pilots(cfg.L, power=1.0, seed=123)
        pilots_torch = torch.from_numpy(pilots).to(device)  # (L,) complex
        
        # Pre-compute all channels and observations on GPU
        N = ant_pos.shape[0]
        S = self.users.shape[0]
        
        t0 = time.time()
        
        # Batch channel computation on GPU
        # Distance: (S, N)
        delta = ant_pos_torch[None, :, :] - users_torch[:, None, :]  # (S, N, 3)
        r = torch.linalg.norm(delta, dim=-1)  # (S, N)
        
        # Phase and amplitude (vectorized on GPU)
        PI2 = 2.0 * torch.pi
        phase = -PI2 * r / self.wavelength
        amp = 1.0 / torch.clamp(r, min=1e-6)
        h_batch = amp * torch.exp(1j * phase)  # (S, N) complex
        
        # Pilot observations (batch)
        Y_batch = h_batch[:, :, None] * pilots_torch[None, None, :]  # (S, N, L)
        
        # Add noise
        if self.noise_var > 0.0:
            noise = (torch.randn(S, N, cfg.L, device=device) * np.sqrt(self.noise_var / 2) +
                     1j * torch.randn(S, N, cfg.L, device=device) * np.sqrt(self.noise_var / 2))
            Y_batch = Y_batch + noise
        
        # Convert to real/imag features: (S, 2*N*L)
        feats_batch = torch.cat([Y_batch.real, Y_batch.imag], dim=1).reshape(S, -1)
        
        # Convert channels to real/imag targets: (S, 2*N)
        targets_batch = torch.cat([h_batch.real, h_batch.imag], dim=1)
        
        # Antenna positions (shared, (N, 2))
        ant_xy = torch.from_numpy(ant_pos[:, :2]).float().to(device)
        
        # Store as GPU tensors
        self.feats = feats_batch
        self.targets = targets_batch
        self.ant_xy = ant_xy  # shared
        self.users_torch = users_torch
        
        dt = time.time() - t0
        print(f"Dataset pre-computation done in {dt:.1f}s on {device}")

    def __len__(self):
        return self.feats.shape[0]

    def __getitem__(self, idx):
        return (
            self.feats[idx],
            self.targets[idx],
            self.ant_xy,
            self.users_torch[idx],
        )


def train(cfg: Config):
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    train_ds = Dataset(cfg, seed=42)
    val_ds = Dataset(cfg, seed=999)
    
    n_ant = cfg.nx * cfg.ny
    net = ChannelEstimator(n_ant=n_ant, L=cfg.L, width=cfg.width, depth=cfg.depth, K=cfg.K).to(device)
    
    # torch.compile for 1.5-2x speedup (PyTorch 2.0+)
    if cfg.use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        net = torch.compile(net)
    
    opt = optim.Adam(net.parameters(), lr=cfg.lr)
    
    # Mixed precision scaler
    scaler = GradScaler(device.type) if cfg.use_amp and device.type == "cuda" else None
    
    # Learning rate scheduler
    if cfg.lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    else:
        scheduler = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=False  # pin_memory=False since data is already on GPU
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, pin_memory=False  # pin_memory=False since data is already on GPU
    )

    def split(vec):
        N = n_ant
        real = vec[:, :N]
        imag = vec[:, N:]
        return real, imag

    # Adaptive loss weighting
    loss_weights = {"mse": 1.0, "phase": 0.3, "amp": 0.3, "smooth": 0.05}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(cfg.epochs):
        # ===== Training =====
        net.train()
        train_loss_total = 0.0
        t0 = time.time()
        
        for feats, target, ant_xy, user in train_loader:
            feats = feats.to(device)
            target = target.to(device)
            ant_xy = ant_xy.to(device)
            user = user.to(device)

            # Mixed precision forward pass
            with autocast(device_type=device.type, enabled=cfg.use_amp):
                pred = net(feats)
                pr, pi = split(pred)
                tr, ti = split(target)

                loss = physics_informed_loss(
                    pr, pi, tr, ti, ant_xy, user, cfg.wavelength, train_ds.shape,
                    w_mse=loss_weights["mse"],
                    w_phase=loss_weights["phase"],
                    w_amp=loss_weights["amp"],
                    w_smooth=loss_weights["smooth"]
                )

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            train_loss_total += loss.item() * feats.shape[0]
        
        train_loss_avg = train_loss_total / len(train_ds)
        
        # ===== Validation (less frequent) =====
        should_validate = (epoch % cfg.val_interval == 0) or (epoch == cfg.epochs - 1)
        if should_validate:
            net.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for feats, target, ant_xy, user in val_loader:
                    feats = feats.to(device)
                    target = target.to(device)
                    ant_xy = ant_xy.to(device)
                    user = user.to(device)

                    pred = net(feats)
                    pr, pi = split(pred)
                    tr, ti = split(target)

                    loss = physics_informed_loss(
                        pr, pi, tr, ti, ant_xy, user, cfg.wavelength, val_ds.shape,
                        w_mse=loss_weights["mse"],
                        w_phase=loss_weights["phase"],
                        w_amp=loss_weights["amp"],
                        w_smooth=loss_weights["smooth"]
                    )
                    val_loss_total += loss.item() * feats.shape[0]
            
            val_loss_avg = val_loss_total / len(val_ds)
        else:
            val_loss_avg = best_val_loss  # use last known
        
        dt = time.time() - t0
        
        # Learning rate schedule step
        if scheduler:
            scheduler.step()
        
        # Adaptive loss weighting: boost phase/amplitude if MSE dominates
        mse_rel = train_loss_avg / (1.0 + train_loss_avg)  # heuristic relevance
        if epoch % 10 == 0 and epoch > 0:
            loss_weights["phase"] = min(0.5, loss_weights["phase"] * 1.05)
            loss_weights["amp"] = min(0.5, loss_weights["amp"] * 1.05)
        
        # Early stopping (only on validation epochs)
        if should_validate and val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        elif should_validate:
            patience_counter += 1
        
        val_str = f"val_loss {val_loss_avg:.4f}" if should_validate else "(skipped)"
        print(f"E{epoch+1:2d} | train_loss {train_loss_avg:.4f} | {val_str} | lr {opt.param_groups[0]['lr']:.2e} | {dt:.1f}s")
        
        if patience_counter >= cfg.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        net.load_state_dict(best_state)
    
    # Final validation NMSE
    net.eval()
    with torch.no_grad():
        feats, target, ant_xy, user = val_ds[0]
        pred = net(feats[None, :].to(device))
        nmse_val = nmse(pred.cpu(), target[None, :])
    print(f"Final Val NMSE (sample): {nmse_val:.4f}")
    return net, train_ds, val_ds


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
