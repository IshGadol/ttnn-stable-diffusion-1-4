from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class OutputLayout:
    date: str
    suite_id: str
    item_id: str

    @property
    def run_dir(self) -> Path:
        return Path("reports") / "ttnn_outputs" / self.date / self.suite_id / "runs" / self.item_id


def write_final_latent(layout: OutputLayout, latent_x0: torch.Tensor) -> Path:
    out_dir = layout.run_dir
    ensure_dir(out_dir)
    path = out_dir / "final_latent_x0.pt"
    torch.save(latent_x0.detach().to("cpu"), path)
    return path
