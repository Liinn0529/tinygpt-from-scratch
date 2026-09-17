import sys
from pathlib import Path

from tinygpt.__main__ import main


def test_smoke_main_prints_expected_fields(tmp_path: Path, monkeypatch, capsys) -> None:
    config = tmp_path / "debug.yaml"
    config.write_text("model:\n  d_model: 64\ntraining:\n  seed: 42\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["tinygpt", "--config", str(config), "--seed", "7"])

    main()

    output = capsys.readouterr().out
    assert "TinyGPT 0.1.0" in output
    assert "device=" in output
    assert "sections=model,training" in output
