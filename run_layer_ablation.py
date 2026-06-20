from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from src.training.trainer import evaluate_checkpoint, train_experiment
from src.training.utils import load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wav2vec2 hidden-layer ablation.")
    parser.add_argument("--config", default="configs/wav2vec2_layer_ablation.yaml")
    parser.add_argument("--eval-split", default="test-clean")
    args = parser.parse_args()

    base_config = load_config(args.config)
    layers = base_config.get("ablation", {}).get("layers", [3, 6, 9, 12])
    root_output = Path(base_config["experiment"]["output_dir"])
    results = {}

    for layer in layers:
        config = deepcopy(base_config)
        config["model"]["hidden_layer"] = int(layer)
        config["experiment"]["name"] = f"{base_config['experiment']['name']}_layer_{layer}"
        config["experiment"]["output_dir"] = str(root_output / f"layer_{layer}")

        output_dir = Path(config["experiment"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        resolved_config_path = output_dir / "resolved_config.yaml"
        with resolved_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

        train_result = train_experiment(config, resolved_config_path)
        test_metrics = evaluate_checkpoint(config, train_result["best_checkpoint"], args.eval_split)
        results[str(layer)] = {
            "best_checkpoint": train_result["best_checkpoint"],
            "dev": train_result["history"][-1]["dev"],
            args.eval_split: test_metrics,
        }

    save_json(root_output / "layer_ablation_results.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

