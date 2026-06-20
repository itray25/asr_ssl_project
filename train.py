from __future__ import annotations

import argparse
import json

from src.training.trainer import train_experiment
from src.training.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a low-resource ASR CTC system.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_experiment(config, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

