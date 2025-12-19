import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineBasis(nn.Module):
    """Cubic B-spline basis over [xmin, xmax] with K knots.
    Implements Cox–de Boor recursion. For efficiency, we precompute uniform knots.
    """
    def __init__(self, xmin: float, xmax: float, K: int, degree: int = 3):
        super().__init__()
        assert degree >= 1
        assert K >= degree + 1
        self.xmin = xmin
        self.xmax = xmax
        self.degree = degree
        # Uniform open knot vector
        # number of basis functions M = K - degree - 1
        # create knots with multiplicity degree+1 at ends
        t = torch.linspace(xmin, xmax, K - 2 * degree)
        t = F.pad(t, (degree, degree), value=xmin)
        t[-degree:] = xmax
        self.register_buffer("knots", t)
        self.M = self.knots.numel() - degree - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., ) -> Vectorized Cox-de Boor recursion
        x = x.clamp(min=self.xmin, max=self.xmax)
        x_shape = x.shape
        x_flat = x.reshape(-1, 1)  # (B, 1) for broadcasting
        t = self.knots
        p = self.degree
        M = self.M
        
        # Initialize: N_{i,0}(x) = 1 if t_i <= x < t_{i+1}, vectorized
        left = t[:M].reshape(1, -1)  # (1, M)
        right = t[1:M+1].reshape(1, -1)  # (1, M)
        N = ((x_flat >= left) & (x_flat < right)).float()  # (B, M)
        N[:, -1] = N[:, -1] | (x_flat.squeeze(-1) == t[-1])  # right endpoint
        
        # Recursion: fully vectorized over all basis functions simultaneously
        for k in range(1, p + 1):
            # Denominators for all i at once
            left_denom = (t[k:k+M] - t[:M]).clamp(min=1e-12)  # (M,)
            right_denom = (t[k+1:k+1+M] - t[1:M+1]).clamp(min=1e-12)  # (M,)
            
            # Left and right contributions: (B, M)
            left_term = ((x_flat - t[:M]) / left_denom.unsqueeze(0)) * N
            right_term_full = ((t[k+1:k+1+M] - x_flat) / right_denom.unsqueeze(0)) * N
            
            # Shift right term and combine
            N = left_term.clone()
            N[:, :-1] = N[:, :-1] + right_term_full[:, 1:]
        
        basis = N.reshape(*x_shape, M)
        return basis


class SplineActivation(nn.Module):
    """Learnable spline activation g(x) = sum c_i B_i(x) + ax + b.
    """
    def __init__(self, xmin: float = -1.0, xmax: float = 1.0, K: int = 16, degree: int = 3):
        super().__init__()
        self.basis = BSplineBasis(xmin, xmax, K, degree)
        self.coeffs = nn.Parameter(torch.zeros(self.basis.M))
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = self.basis(x)  # (..., M)
        g = (B * self.coeffs).sum(dim=-1) + self.a * x + self.b
        return g


class KANLayer(nn.Module):
    """KAN layer: y_j = sum_i g_{ij}(x_i) + linear mix.
    Uses per-input spline activations per output unit, plus optional linear residual.
    """
    def __init__(self, in_dim: int, out_dim: int, K: int = 16, degree: int = 3, xmin: float = -1.0, xmax: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        activations = []
        for j in range(out_dim):
            row = nn.ModuleList([SplineActivation(xmin, xmax, K, degree) for _ in range(in_dim)])
            activations.append(row)
        self.activations = nn.ModuleList(activations)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        comps = []
        for j in range(self.out_dim):
            gsum = 0.0
            for i in range(self.in_dim):
                gsum = gsum + self.activations[j][i](x[:, i])
            comps.append(gsum)
        G = torch.stack(comps, dim=-1)
        return G + self.linear(x)


class KAN(nn.Module):
    """Simple multi-layer KAN network."""
    def __init__(self, dims, K: int = 16, degree: int = 3, xmin: float = -1.0, xmax: float = 1.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(KANLayer(dims[i], dims[i + 1], K=K, degree=degree, xmin=xmin, xmax=xmax))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.tanh(x)  # smooth nonlinearity between KAN layers
        x = self.layers[-1](x)
        return x


class ChannelEstimator(nn.Module):
    """KAN-based channel estimator mapping pilot features to complex channel vector.

    Input: real/imag pilot observations flattened (2*N*L)
    Output: real/imag channel vector flattened (2*N)
    """
    def __init__(self, n_ant: int, L: int, width: int = 256, depth: int = 2, K: int = 16):
        super().__init__()
        in_dim = 2 * n_ant * L
        out_dim = 2 * n_ant
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        self.net = KAN(dims, K=K, degree=3, xmin=-2.0, xmax=2.0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)

    @staticmethod
    def split_complex(vec: torch.Tensor):
        N2 = vec.shape[-1]
        N = N2 // 2
        real = vec[..., :N]
        imag = vec[..., N:]
        return real, imag

    @staticmethod
    def combine_complex(real: torch.Tensor, imag: torch.Tensor):
        return torch.complex(real, imag)
