from __future__ import annotations

import argparse
from pathlib import Path

from tinygpt import __version__
from tinygpt.config import load_config
from tinygpt.utils import get_device, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TinyGPT project smoke check")
    parser.add_argument("--config", type=Path, default=Path("configs/debug.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    config = load_config(args.config)

    print(f"TinyGPT {__version__}")
    print(f"device={get_device().type}")
    print(f"config={args.config}")
    print(f"sections={','.join(sorted(config))}")


if __name__ == "__main__":
    main()
