"""Acceptance checks for M0 project setup."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_hardware_check_reports_python_and_cuda() -> None:
    """The hardware script prints both required setup facts."""
    script = Path(__file__).parents[1] / "tools" / "check_hardware.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Python:" in result.stdout
    assert "CUDA GPU available:" in result.stdout
