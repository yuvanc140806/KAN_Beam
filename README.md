# KAN-Beam: Physics-Aligned Near-Field Channel Estimation & Beamforming

KAN-Beam is a compact, physics-informed deep learning framework for near-field (Fresnel) holographic MIMO in 6G. It couples a first-principles near-field simulator with a Kolmogorov–Arnold Network (KAN) estimator and beamforming controller. The focus is structural alignment with electromagnetic physics, not brute-force fitting.

## Key Ideas
- Near-field channels: `h_n = (1/r_n) * exp(-j 2π r_n / λ)` computed exactly from geometry.
- Inverse via pilots: recover high-dimensional spherical-wave channel from noisy pilot observations.
- KAN estimator: spline-based learnable activations approximate smooth oscillatory physics efficiently.
- Physics-informed loss: constraints on phase curvature, amplitude decay, and spatial smoothness.
- Beamforming: unit-modulus phase control to focus energy at a spatial point without iterative solvers.

## Quick Start

### 1) Install dependencies

```powershell
cd "c:\Users\C. Tinesh Karthick\Documents\KAN_Beam"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Train a small model

```powershell
python train.py
```

### 3) Evaluate generalization & beam focusing

```powershell
python eval.py
```

## Repository Structure
- `kan_beam/physics.py`: Geometry → exact near-field channels and beampatterns.
- `kan_beam/pilots.py`: Pilot generation and observation model.
- `kan_beam/kan.py`: Minimal KAN implementation and `ChannelEstimator`.
- `kan_beam/losses.py`: Physics-informed loss components.
- `kan_beam/beamforming.py`: Phase control from estimated channels.
- `train.py`: Dataset synthesis and training loop.
- `eval.py`: Generalization NMSE and focusing peak metrics.

## Configuration Notes
Default config trains on an 8×8 aperture with 4 pilots for ~10 epochs on CPU. Adjust `Config` in `train.py` for aperture size, pilots, noise, and network width/depth. Wavelength defaults to 0.1 m (≈3 GHz); change per scenario.

## Metrics & Philosophy
- NMSE on channels and beampattern peak at focus.
- Parameter efficiency vs CNNs (KAN achieves comparable accuracy with fewer parameters).
- Stability under noise and geometry shifts (evaluate with `eval.py`).

## Scientific Contribution
Unifies near-field physics, Kolmogorov–Arnold function decomposition, physics-informed learning, and neural beamforming into a single, executable framework.
