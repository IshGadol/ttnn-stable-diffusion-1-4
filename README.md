# ttnn-stable-diffusion-1-4

Bring up Stable Diffusion 1.4 (512x512) on Tenstorrent N300/N150 using TTNN.

## scope

- Define a clean environment and dependency stack for SD 1.4 on Tenstorrent.
- Start from an existing CPU baseline (diffusers / PyTorch) and port the core
  UNet + attention blocks to TTNN.
- Produce correctness checks vs. CPU baseline for a fixed set of prompts.
- Measure throughput and latency on N300 (and N150 if available).
- Provide clear, reproducible instructions for bringing up SD 1.4 on TTNN.

## status (2025-12-12)

- Repo skeleton created.
- Bounty target: “[Bounty $1500] Add model: Stable Diffusion 1.4 (512x512) #1041”.
- TTNN environment bring-up and N300 access handled in parallel with the
  Faster-RCNN / TTNN project.
- Next steps:
  - Finalize environment (Python + TTNN + diffusers) once N300 access is active.
  - Implement a simple CPU baseline script for SD 1.4 (single prompt).
  - Define TTNN porting roadmap (similar style to Faster-RCNN roadmap).

## folders

- `src/` — model code, TTNN modules, helpers.
- `scripts/` — CLI scripts for running CPU and TTNN inference.
- `reports/` — logs, perf metrics, and parity reports.
- `configs/` — YAML/JSON configs for prompts, sampling, and model variants.
- `models/` — pointers or scripts to download SD 1.4 weights.

