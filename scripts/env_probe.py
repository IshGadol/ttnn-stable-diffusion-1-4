#!/usr/bin/env python3
"""
env_probe.py

Safe environment probe for the SD1.4 TTNN bounty repo.
- Prints Python + key package versions
- Soft-checks TTNN import + basic device availability
- Does NOT run any model code
"""

from __future__ import annotations

import importlib
import platform
import sys
from datetime import datetime
from typing import Optional


def _get_version(pkg_name: str) -> Optional[str]:
    try:
        mod = importlib.import_module(pkg_name)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def _print_kv(k: str, v: str) -> None:
    print(f"{k:24s}: {v}")


def main() -> int:
    print("=== SD1.4 TTNN Bounty — env probe ===")
    _print_kv("timestamp", datetime.utcnow().isoformat() + "Z")
    _print_kv("platform", platform.platform())
    _print_kv("python", sys.version.replace("\n", " "))

    print("\n--- package versions ---")
    for pkg in ["torch", "diffusers", "transformers", "safetensors", "numpy"]:
        ver = _get_version(pkg)
        _print_kv(pkg, ver if ver else "NOT INSTALLED")

    print("\n--- torch details ---")
    try:
        import torch

        _print_kv("torch.cuda.is_available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            _print_kv("torch.version.cuda", str(torch.version.cuda))
            _print_kv("cuda.device_count", str(torch.cuda.device_count()))
            _print_kv("cuda.device_0", torch.cuda.get_device_name(0))
    except Exception as e:
        _print_kv("torch import", f"FAILED ({type(e).__name__}: {e})")

    print("\n--- ttnn probe (soft) ---")
    try:
        import ttnn  # type: ignore

        _print_kv("ttnn import", "OK")
        _print_kv("ttnn.__version__", getattr(ttnn, "__version__", "UNKNOWN"))

        # Soft device check: API varies by version; don't hard-fail.
        device_info = "UNKNOWN"
        try:
            # Some versions expose ttnn.get_device_ids() or similar.
            if hasattr(ttnn, "get_device_ids"):
                device_info = str(ttnn.get_device_ids())
            elif hasattr(ttnn, "device"):
                device_info = "ttnn.device present"
            else:
                device_info = "no known device query API"
        except Exception as e:
            device_info = f"device query failed ({type(e).__name__}: {e})"

        _print_kv("ttnn devices", device_info)

    except Exception as e:
        _print_kv("ttnn import", f"NOT AVAILABLE ({type(e).__name__}: {e})")

    print("\n=== probe complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
