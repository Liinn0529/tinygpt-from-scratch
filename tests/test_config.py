from pathlib import Path

import pytest

from tinygpt.config import load_config


def test_load_config_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  d_model: 128\ntraining:\n  seed: 42\n", encoding="utf-8")

    config = load_config(path)

    assert config["model"]["d_model"] == 128
    assert config["training"]["seed"] == 42


def test_load_config_empty_document_returns_empty_dict(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == {}


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("definitely-does-not-exist.yaml")
