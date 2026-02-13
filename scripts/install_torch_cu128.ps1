Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Removing existing torch packages..."
python -m pip uninstall -y torch torchvision torchaudio

Write-Host "Installing stable PyTorch CUDA 12.8 wheels..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host ""
Write-Host "Verify with: python scripts/verify_torch_cuda.py"
