import random

import numpy as np
import torch

from tinygpt.utils import get_device, seed_everything


def _draw_rng_sample() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4)


def test_seed_everything_is_reproducible() -> None:
    seed_everything(1234)
    first = _draw_rng_sample()

    seed_everything(1234)
    second = _draw_rng_sample()

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_get_device_returns_supported_device() -> None:
    assert get_device().type in {"cpu", "cuda", "mps"}
