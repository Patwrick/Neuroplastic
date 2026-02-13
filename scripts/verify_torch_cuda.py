from __future__ import annotations

import torch


def main() -> None:
    cuda_available = torch.cuda.is_available()
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"cuda_device_0: {device_name}")
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = x * 2.0 + 1.0
        print(f"cuda_tensor_check: {y.tolist()}")
        return

    print("CUDA is not available in this Python environment.")
    print("Install CUDA wheels with one of these scripts, then re-run this verifier:")
    print(r".\scripts\install_torch_cu128.ps1")
    print(r".\scripts\install_torch_nightly_cu128.ps1")


if __name__ == "__main__":
    main()
