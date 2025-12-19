Param(
    [int]$Epochs = 5,
    [int]$Nx = 8,
    [int]$Ny = 8
)

$env:PYTHONUNBUFFERED = "1"
python - << 'PY'
from train import Config, train

cfg = Config(epochs=$Epochs, nx=$Nx, ny=$Ny)
train(cfg)
PY
