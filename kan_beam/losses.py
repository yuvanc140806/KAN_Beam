import torch
import torch.nn.functional as F


def mse_loss(pred_real, pred_imag, true_real, true_imag):
    return F.mse_loss(pred_real, true_real) + F.mse_loss(pred_imag, true_imag)


def phase_curvature_loss(pred_real, pred_imag, ant_xy, user_pos, wavelength):
    """Encourage spherical-wave phase relative to geometry (batched).
    ant_xy: (B, N, 2) or (N, 2)
    user_pos: (B, 3) or (3,)
    """
    device = pred_real.device
    B, N = pred_real.shape

    # Ensure batch dimensions
    ant_xy = ant_xy.to(device)
    if ant_xy.dim() == 2:
        ant_xy = ant_xy.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

    user_pos = user_pos.to(device)
    if user_pos.dim() == 1:
        user_pos = user_pos.unsqueeze(0).expand(B, -1)  # (B, 3)

    z_ant = torch.zeros((B, N, 1), device=device)
    ant_pos = torch.cat([ant_xy, z_ant], dim=-1)  # (B, N, 3)
    delta = ant_pos - user_pos[:, None, :]  # (B, N, 3)
    r = torch.linalg.norm(delta, dim=-1)  # (B, N)
    true_phase = -2.0 * torch.pi * r / wavelength
    pred_phase = torch.atan2(pred_imag, pred_real)

    # Remove per-sample global phase offset
    diff = pred_phase - true_phase
    offset = diff.mean(dim=1, keepdim=True)
    aligned = diff - offset
    return torch.mean(aligned ** 2)


def amplitude_decay_loss(pred_real, pred_imag, ant_xy, user_pos, wavelength):
    device = pred_real.device
    B, N = pred_real.shape

    ant_xy = ant_xy.to(device)
    if ant_xy.dim() == 2:
        ant_xy = ant_xy.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

    user_pos = user_pos.to(device)
    if user_pos.dim() == 1:
        user_pos = user_pos.unsqueeze(0).expand(B, -1)  # (B, 3)

    z_ant = torch.zeros((B, N, 1), device=device)
    ant_pos = torch.cat([ant_xy, z_ant], dim=-1)  # (B, N, 3)
    delta = ant_pos - user_pos[:, None, :]
    r = torch.linalg.norm(delta, dim=-1)  # (B, N)

    target_amp = 1.0 / torch.clamp(r, min=1e-6)
    pred_amp = torch.sqrt(pred_real ** 2 + pred_imag ** 2)
    # Scale-invariant comparison (normalize both per-sample)
    target_norm = target_amp / (target_amp.mean(dim=1, keepdim=True) + 1e-9)
    pred_norm = pred_amp / (pred_amp.mean(dim=1, keepdim=True) + 1e-9)
    return torch.mean((pred_norm - target_norm) ** 2)


def spatial_smoothness_loss(pred_real, pred_imag, shape):
    """Total variation across the aperture grid (batched)."""
    nx, ny = shape
    B = pred_real.shape[0]
    real = pred_real.reshape(B, nx, ny)
    imag = pred_imag.reshape(B, nx, ny)
    tv = 0.0
    tv += torch.mean(torch.abs(real[:, 1:, :] - real[:, :-1, :]))
    tv += torch.mean(torch.abs(real[:, :, 1:] - real[:, :, :-1]))
    tv += torch.mean(torch.abs(imag[:, 1:, :] - imag[:, :-1, :]))
    tv += torch.mean(torch.abs(imag[:, :, 1:] - imag[:, :, :-1]))
    return tv


def compute_adaptive_weights(mse_val, phase_val, amp_val, smooth_val, target_ratio=0.5):
    """Dynamically balance loss terms using gradient magnitude heuristic.
    
    Ensures physics constraints don't get overwhelmed by MSE scale.
    target_ratio: desired ratio of physics losses to MSE (e.g., 0.5 = half the MSE magnitude)
    """
    physics_mag = phase_val.detach() + amp_val.detach() + smooth_val.detach()
    mse_mag = mse_val.detach()
    
    # Prevent division by zero
    if mse_mag < 1e-8 or physics_mag < 1e-8:
        return 1.0, 1.0, 1.0, 1.0
    
    # Scale physics weights to maintain desired ratio
    scale = (target_ratio * mse_mag) / (physics_mag + 1e-9)
    w_phase = min(1.0, scale * 0.4)
    w_amp = min(1.0, scale * 0.4)
    w_smooth = min(0.2, scale * 0.05)
    
    return 1.0, w_phase, w_amp, w_smooth


def physics_informed_loss(pred_real, pred_imag, true_real, true_imag, ant_xy, user_pos, wavelength, shape,
                           w_mse=1.0, w_phase=0.5, w_amp=0.5, w_smooth=0.1, adaptive=False, target_ratio=0.5):
    """Physics-informed composite loss.
    
    Args:
        adaptive: if True, reweight physics terms dynamically
        target_ratio: physics-to-MSE magnitude ratio (used if adaptive=True)
    """
    loss_mse = mse_loss(pred_real, pred_imag, true_real, true_imag)
    loss_phase = phase_curvature_loss(pred_real, pred_imag, ant_xy, user_pos, wavelength)
    loss_amp = amplitude_decay_loss(pred_real, pred_imag, ant_xy, user_pos, wavelength)
    loss_smooth = spatial_smoothness_loss(pred_real, pred_imag, shape)
    
    if adaptive:
        w_mse, w_phase, w_amp, w_smooth = compute_adaptive_weights(
            loss_mse, loss_phase, loss_amp, loss_smooth, target_ratio=target_ratio
        )
    
    total_loss = (w_mse * loss_mse + 
                  w_phase * loss_phase + 
                  w_amp * loss_amp + 
                  w_smooth * loss_smooth)
    return total_loss
