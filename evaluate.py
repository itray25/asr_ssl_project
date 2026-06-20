from __future__ import annotations

import argparse
import json

from src.training.trainer import evaluate_checkpoint
from src.training.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an ASR checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--split", default="test-clean", help="LibriSpeech split or dummy split name.")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = evaluate_checkpoint(config, args.checkpoint, args.split)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

