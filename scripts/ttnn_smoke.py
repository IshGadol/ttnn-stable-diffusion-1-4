#!/usr/bin/env python3
from __future__ import annotations
"""
TTNN smoke test (safe-by-default).

Default behavior:
- CPU-only
- No device probing
- Subprocess + timeout isolation

Device access is opt-in via --allow-device.
"""


import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

REPORT_DIR = Path("reports/ttnn_smoke")


@dataclass
class SmokeResult:
    ok: bool
    mode: str  # "cpu_only" | "allow_device"
    elapsed_sec: float
    python_exe: str
    python_version: str
    platform: Dict[str, str]
    env: Dict[str, str]
    ttnn_import_ok: bool
    ttnn_version: str
    details: Dict[str, Any]
    error: str


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _run_subprocess(payload: Dict[str, Any], timeout_sec: int) -> Tuple[int, str, str]:
    """
    Run the TTNN probe in a separate Python process so that if it hangs or
    segfaults, the parent can timeout / report cleanly.
    """
    code = r"""
import json, os, sys, time, traceback

payload = json.loads(sys.stdin.read())
allow_device = bool(payload.get("allow_device", False))

# Safety knobs (best-effort).
os.environ.setdefault("TTNN_SMOKE", "1")
os.environ.setdefault("TTNN_LOG_LEVEL", os.environ.get("TTNN_LOG_LEVEL", "WARN"))

# CPU-only guardrails unless explicitly allowed to touch devices.
if not allow_device:
    os.environ.setdefault("TTNN_BACKEND", "cpu")
    os.environ.setdefault("TTNN_DEVICE", "none")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

out = {
  "ttnn_import_ok": False,
  "ttnn_version": "(unknown)",
  "device_probe": {"attempted": False, "ok": False, "note": ""},
  "timing": {},
  "env_effective": {k: os.environ.get(k, "") for k in ["TTNN_BACKEND","TTNN_DEVICE","CUDA_VISIBLE_DEVICES","TTNN_LOG_LEVEL","TTNN_SMOKE"]},
}

t0 = time.time()
try:
    import ttnn  # noqa: F401
    out["ttnn_import_ok"] = True
    out["ttnn_version"] = getattr(ttnn, "__version__", "(none)")
except Exception as e:
    out["error"] = f"import_error: {e!r}"
    out["traceback"] = traceback.format_exc()
    out["timing"]["import_sec"] = round(time.time() - t0, 6)
    print(json.dumps(out, indent=2))
    sys.exit(2)

out["timing"]["import_sec"] = round(time.time() - t0, 6)

# Optional device probe path (opt-in only).
if allow_device:
    out["device_probe"]["attempted"] = True
    try:
        import ttnn

        note_parts = []
        opened = False

        if hasattr(ttnn, "open_device") and callable(getattr(ttnn, "open_device")):
            dev = ttnn.open_device(device_id=0)  # type: ignore[arg-type]
            opened = True
            note_parts.append("used ttnn.open_device(device_id=0)")

            if hasattr(ttnn, "close_device") and callable(getattr(ttnn, "close_device")):
                ttnn.close_device(dev)
                note_parts.append("used ttnn.close_device(dev)")
            elif hasattr(dev, "close") and callable(getattr(dev, "close")):
                dev.close()
                note_parts.append("used dev.close()")
            else:
                note_parts.append("no close method found; leaving device object to GC")
        else:
            note_parts.append("no recognized open_device API; skipped device open/close")

        out["device_probe"]["ok"] = True
        out["device_probe"]["note"] = "; ".join(note_parts)

    except Exception as e:
        out["device_probe"]["ok"] = False
        out["device_probe"]["note"] = f"device_probe_error: {e!r}"
        out["traceback"] = traceback.format_exc()
        print(json.dumps(out, indent=2))
        sys.exit(3)

print(json.dumps(out, indent=2))
sys.exit(0)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    try:
        stdout, stderr = proc.communicate(json.dumps(payload), timeout=timeout_sec)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        return 124, stdout, stderr


def _extract_last_json_object(mixed_text: str) -> dict:
    """
    TTNN/UMD can emit log lines to stdout before/after the JSON payload.
    Logs may contain brace fragments like "{0}" that are NOT JSON.
    We locate a JSON object that includes the sentinel key "ttnn_import_ok".
    """
    import json as _json

    text = mixed_text.strip()
    if not text:
        return {}

    # Greedy scan for a JSON object that contains the sentinel key.
    # We search for the last occurrence of the sentinel and then expand outward.
    sentinel = '"ttnn_import_ok"'
    idx = text.rfind(sentinel)
    if idx == -1:
        return {}

    # Find the nearest '{' before the sentinel
    start = text.rfind("{", 0, idx)
    if start == -1:
        return {}

    candidate = text[start:]

    # Trim candidate to the last '}' to avoid trailing logs
    end = candidate.rfind("}")
    if end == -1:
        return {}
    candidate = candidate[: end + 1]

    try:
        return _json.loads(candidate)
    except Exception:
        # As a fallback, try to extract the final {...} block that contains the sentinel.
        # This is more expensive but safer than mis-parsing "{0}" fragments.
        blocks = re.findall(r"\{[\s\S]*?\}", text)
        for blk in reversed(blocks):
            if sentinel in blk:
                try:
                    return _json.loads(blk)
                except Exception:
                    continue
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-device", action="store_true", help="Opt-in to device probing/open/close attempts.")
    ap.add_argument("--require-ttnn", action="store_true", help="Fail if ttnn is not importable (use on real TTNN env).")
    ap.add_argument("--timeout-sec", type=int, default=20, help="Hard timeout for TTNN probe subprocess.")
    ap.add_argument("--outdir", type=str, default=str(REPORT_DIR), help="Report directory (default: reports/ttnn_smoke).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mode = "allow_device" if args.allow_device else "cpu_only"
    ts = _now_ts()
    report_json = outdir / f"ttnn_smoke_{mode}_{ts}.json"
    report_txt = outdir / f"ttnn_smoke_{mode}_{ts}.txt"

    env_snapshot = {
        # Keep this minimal; do NOT dump secrets.
        "TTNN_BACKEND": os.environ.get("TTNN_BACKEND", ""),
        "TTNN_DEVICE": os.environ.get("TTNN_DEVICE", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
    }

    t0 = time.time()
    result = SmokeResult(
        ok=False,
        mode=mode,
        elapsed_sec=0.0,
        python_exe=sys.executable,
        python_version=sys.version.replace("\n", " "),
        platform={
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        env=env_snapshot,
        ttnn_import_ok=False,
        ttnn_version="(unknown)",
        details={},
        error="",
    )

    payload = {"allow_device": bool(args.allow_device)}
    try:
        rc, stdout, stderr = _run_subprocess(payload, timeout_sec=args.timeout_sec)
        result.details["probe_returncode"] = rc
        result.details["probe_stdout"] = stdout.strip()
        result.details["probe_stderr"] = stderr.strip()

        if rc == 124:
            result.error = f"timeout: TTNN probe exceeded {args.timeout_sec}s"
        elif rc != 0:
            # rc=2 is import failure (including ModuleNotFoundError)
            if rc == 2 and (not args.require_ttnn):
                result.ok = True
                result.error = "SKIP: ttnn not installed (run with --require-ttnn to enforce)"
            else:
                result.error = f"probe_failed: returncode={rc}"
        else:
            try:
                probe = _extract_last_json_object(stdout)
                if not probe and stdout.strip():
                    raise ValueError("no_json_object_found_in_stdout")
            except Exception as e:
                result.error = f"probe_output_parse_error: {e!r}"
                probe = {}

            result.ttnn_import_ok = bool(probe.get("ttnn_import_ok", False))
            result.ttnn_version = str(probe.get("ttnn_version", "(unknown)"))
            result.details["probe"] = probe

            if not result.ttnn_import_ok:
                result.error = "ttnn_import_failed"
            else:
                result.ok = True

    except Exception as e:
        result.error = f"parent_exception: {e!r}"
        result.details["traceback"] = traceback.format_exc()

    result.elapsed_sec = round(time.time() - t0, 6)

    report_json.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    lines = []
    lines.append(f"TTNN SMOKE RESULT: {'PASS' if result.ok else 'FAIL'}")
    lines.append(f"mode: {result.mode}")
    lines.append(f"elapsed_sec: {result.elapsed_sec}")
    lines.append(f"python: {result.python_exe}")
    lines.append(f"python_version: {result.python_version}")
    lines.append(f"ttnn_import_ok: {result.ttnn_import_ok}")
    lines.append(f"ttnn_version: {result.ttnn_version}")
    if result.error:
        lines.append(f"error: {result.error}")
    lines.append(f"report_json: {report_json}")
    lines.append(f"report_txt: {report_txt}")

    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
