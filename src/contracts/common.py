from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


Shape = Tuple[int, ...]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Shape  # use -1 for "any" in that dimension
    dtype: torch.dtype
    notes: str = ""


def _shape_matches(actual: Sequence[int], expected: Sequence[int]) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if e == -1:
            continue
        if a != e:
            return False
    return True


def assert_tensor(t: torch.Tensor, spec: TensorSpec, *, allow_none: bool = False) -> None:
    if t is None:
        if allow_none:
            return
        raise TypeError(f"{spec.name}: tensor is None")

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{spec.name}: expected torch.Tensor, got {type(t)}")

    if not _shape_matches(tuple(t.shape), spec.shape):
        raise ValueError(
            f"{spec.name}: shape mismatch. got {tuple(t.shape)} expected {spec.shape}. {spec.notes}"
        )

    if t.dtype != spec.dtype:
        raise TypeError(
            f"{spec.name}: dtype mismatch. got {t.dtype} expected {spec.dtype}. {spec.notes}"
        )
