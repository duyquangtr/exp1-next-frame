"""Environment smoke test: torch version and CUDA availability."""

import torch


def test_torch_version_is_2_11() -> None:
    version = torch.__version__
    print(f"torch.__version__={version}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    assert version.startswith("2.11"), f"expected torch 2.11.x, got {version}"
