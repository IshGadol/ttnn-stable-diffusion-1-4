# Stable Diffusion 1.4 TTNN Bounty — Status Log

Repository: ttnn-stable-diffusion-1-4
Target: Tenstorrent TTNN Bounty — "[Bounty $1500] Add model: Stable Diffusion 1.4 (512x512) #1041"
Status Date: 2025-12-11

------------------------------------------------------------
1. Environment State
------------------------------------------------------------

Python
- Python 3.12 (.venv on ControlNode)
- Requirements installed via "make env":
  - torch 2.9.1 (CUDA 12.x build)
  - torchvision
  - diffusers
  - transformers
  - safetensors
  - Pillow
  - numpy

Hardware
- Tenstorrent N300/N150 access is pending activation (via Koyeb).
- All work currently executes on ControlNode (CPU environment).

Hugging Face
- User has a valid HF account and can authenticate ("huggingface-cli login").
- Access required for "CompVis/stable-diffusion-v1-4".

------------------------------------------------------------
2. Repo Structure (as of this commit)
------------------------------------------------------------

src/
  models/
  pipelines/
    cpu_sd14_pipeline.py
  schedulers/
  ttnn_impl/
    ttnn_sd14_pipeline.py
scripts/
  run_cpu_sd14.py
  run_ttnn_sd14.py
tests/
  test_api_surface.py
configs/
models/
reports/

All directories are import-ready (they contain __init__.py where needed).

------------------------------------------------------------
3. Completed Work
------------------------------------------------------------

CPU Baseline
- CPUStableDiffusionPipelineWrapper implemented under src/pipelines/cpu_sd14_pipeline.py.
- CLI harness: scripts/run_cpu_sd14.py.
- API surface designed to mirror the TTNN skeleton so higher-level code can swap backends.

TTNN Skeleton
- TTNNStableDiffusionPipeline implemented under src/ttnn_impl/ttnn_sd14_pipeline.py.
- Returns dummy latents with shape (1, 4, 64, 64) for a 512x512 latent grid.
- CLI harness: scripts/run_ttnn_sd14.py.
- Deterministic behavior when a seed is provided.

Tests
- tests/test_api_surface.py:
  - Verifies CPU config defaults.
  - Verifies TTNN config defaults.
  - Verifies TTNN dummy latent shape contract.
- All tests pass via: python -m unittest discover -s tests -p "test_*.py" -v

------------------------------------------------------------
4. Pending / Future Work
------------------------------------------------------------

TTNN Integration
- Real UNet graph port to TTNN.
- VAE decode path on TTNN.
- CLIP text encoder on TTNN.
- TTNN device memory and execution buffers.

Functional Pipeline Items
- Scheduler and sampler implementations compatible with TTNN.
- Parity tests vs CPU baseline for fixed prompts and seeds.
- Logging and reporting for latency, throughput, and memory usage.

Repository Amenities
- Model download helpers under models/.
- Structured inference configs under configs/.
- Reporting and metrics helpers under reports/.

------------------------------------------------------------
5. Notes for Reviewers
------------------------------------------------------------

- Structure intentionally mirrors the Faster-RCNN TTNN bounty project.
- API surfaces for CPU and TTNN backends are now stable and enforced by tests.
- CPU baseline is ready but not exercised here until HF authentication and model license acceptance are set up.
- TTNN skeleton is ready for N300 bring-up as soon as hardware access is available.
