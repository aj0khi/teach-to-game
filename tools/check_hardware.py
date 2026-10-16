"""Report the Python version and whether a CUDA GPU is available."""

from __future__ import annotations

import platform
from importlib import import_module
from importlib.util import find_spec


def cuda_available() -> bool:
    """Return whether the optional PyTorch installation can use CUDA."""
    if find_spec("torch") is None:
        return False
    torch = import_module("torch")
    return bool(torch.cuda.is_available())


def main() -> None:
    """Print the hardware facts needed by the project setup check."""
    print(f"Python: {platform.python_version()}")
    print(f"CUDA GPU available: {'yes' if cuda_available() else 'no'}")


if __name__ == "__main__":
    main()
