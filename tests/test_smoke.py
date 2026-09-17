from tinygpt.utils import get_device, seed_everything


def test_smoke() -> None:
    seed_everything(42)
    assert get_device().type in {"cpu", "cuda", "mps"}
