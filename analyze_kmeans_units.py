from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.librispeech import build_dataset
from src.training.utils import load_config, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze K-means discrete speech units from SSL features.")
    parser.add_argument("--config", required=True, help="Path to an existing ASR config.")
    parser.add_argument("--split", default="dev-clean", help="LibriSpeech split to analyze.")
    parser.add_argument("--k", type=int, required=True, help="K-means codebook size.")
    parser.add_argument("--max-frames", type=int, default=200_000, help="Maximum SSL frames used for K-means.")
    parser.add_argument("--max-utts", type=int, default=None, help="Optional maximum number of utterances to analyze.")
    parser.add_argument("--batch-size", type=int, default=1, help="Feature extraction batch size.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser.parse_args()


def collate_waveforms(items: list[dict]) -> dict:
    waveforms = []
    lengths = []
    ids = []
    for item in items:
        waveform = item["waveform"]
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0) if waveform.size(0) == 1 else waveform.mean(dim=0)
        waveform = waveform.float()
        waveforms.append(waveform)
        lengths.append(waveform.numel())
        ids.append(item["id"])
    padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    return {"waveforms": padded, "waveform_lengths": torch.tensor(lengths, dtype=torch.long), "ids": ids}


def output_lengths(model, waveform_lengths: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_get_feat_extract_output_lengths"):
        return model._get_feat_extract_output_lengths(waveform_lengths).long()
    return torch.div(waveform_lengths, 320, rounding_mode="floor").clamp_min(1)


def extract_hidden_states(model, batch: dict, hidden_layer: int | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    waveforms = batch["waveforms"].to(device)
    waveform_lengths = batch["waveform_lengths"].to(device)
    attention_mask = torch.arange(waveforms.size(1), device=device).unsqueeze(0)
    attention_mask = attention_mask < waveform_lengths.unsqueeze(1)
    outputs = model(waveforms, attention_mask=attention_mask.long(), output_hidden_states=hidden_layer is not None)
    features = outputs.last_hidden_state if hidden_layer is None else outputs.hidden_states[hidden_layer]
    lengths = output_lengths(model, waveform_lengths).clamp_max(features.size(1))
    return features, lengths


def deduplicate(units: list[int]) -> tuple[list[int], list[int]]:
    if not units:
        return [], []
    deduped = [units[0]]
    durations = [1]
    for unit in units[1:]:
        if unit == deduped[-1]:
            durations[-1] += 1
        else:
            deduped.append(unit)
            durations.append(1)
    return deduped, durations


def summarize_sequences(sequences: list[list[int]], lengths: list[int], sample_rate: int, k: int) -> dict:
    total_frames = sum(len(seq) for seq in sequences)
    total_seconds = sum(lengths) / float(sample_rate)
    unit_counts = [0] * k
    dedup_lengths = []
    run_lengths = []

    for seq in sequences:
        for unit in seq:
            unit_counts[unit] += 1
        deduped, durations = deduplicate(seq)
        dedup_lengths.append(len(deduped))
        run_lengths.extend(durations)

    token_rate = total_frames / total_seconds if total_seconds > 0 else 0.0
    dedup_total = sum(dedup_lengths)
    dedup_token_rate = dedup_total / total_seconds if total_seconds > 0 else 0.0
    bits_per_token = math.log2(k)
    probs = [count / total_frames if total_frames else 0.0 for count in unit_counts]
    entropy = -sum(prob * math.log2(prob) for prob in probs if prob > 0)

    return {
        "num_utterances": len(sequences),
        "total_seconds": total_seconds,
        "total_tokens": total_frames,
        "token_rate": token_rate,
        "dedup_total_tokens": dedup_total,
        "dedup_token_rate": dedup_token_rate,
        "bitrate": token_rate * bits_per_token,
        "dedup_bitrate": dedup_token_rate * bits_per_token,
        "avg_tokens_per_utterance": total_frames / len(sequences) if sequences else 0.0,
        "avg_dedup_tokens_per_utterance": dedup_total / len(sequences) if sequences else 0.0,
        "dedup_compression_ratio": total_frames / dedup_total if dedup_total else 0.0,
        "unit_counts": unit_counts,
        "unit_probs": probs,
        "used_units": sum(1 for count in unit_counts if count > 0),
        "entropy_bits": entropy,
        "effective_num_units": 2**entropy,
        "avg_run_length_frames": sum(run_lengths) / len(run_lengths) if run_lengths else 0.0,
        "max_run_length_frames": max(run_lengths) if run_lengths else 0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("experiment", {}).get("seed", 1337)))

    try:
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("numpy, scikit-learn, and transformers are required for K-means unit analysis.") from exc

    data_cfg = config["data"]
    model_cfg = config["model"]
    device = resolve_device(config["training"].get("device", "auto"))
    dataset = build_dataset(config, args.split, train=False)
    if args.max_utts is not None:
        dataset = Subset(dataset, range(min(args.max_utts, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_waveforms)

    model = AutoModel.from_pretrained(model_cfg["ssl_model_name"]).to(device)
    model.eval()
    hidden_layer = model_cfg.get("hidden_layer")

    sampled_features = []
    sequences = []
    waveform_lengths = []
    remaining_frames = args.max_frames

    with torch.no_grad():
        for batch in tqdm(loader, desc="extract ssl units"):
            features, feature_lengths = extract_hidden_states(model, batch, hidden_layer, device)
            features_cpu = features.cpu()
            feature_lengths_cpu = feature_lengths.cpu().tolist()
            waveform_lengths.extend(batch["waveform_lengths"].tolist())

            for batch_idx, feature_length in enumerate(feature_lengths_cpu):
                utterance_features = features_cpu[batch_idx, :feature_length]
                if remaining_frames > 0:
                    take = min(remaining_frames, utterance_features.size(0))
                    sampled_features.append(utterance_features[:take])
                    remaining_frames -= take
                if remaining_frames == 0:
                    break
            if remaining_frames == 0:
                break

    if not sampled_features:
        raise RuntimeError("No SSL frames were extracted for K-means.")

    train_features = torch.cat(sampled_features, dim=0).numpy().astype(np.float32)
    kmeans = MiniBatchKMeans(n_clusters=args.k, batch_size=4096, n_init="auto", random_state=1337)
    kmeans.fit(train_features)

    waveform_lengths = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict units"):
            features, feature_lengths = extract_hidden_states(model, batch, hidden_layer, device)
            features_cpu = features.cpu()
            feature_lengths_cpu = feature_lengths.cpu().tolist()
            waveform_lengths.extend(batch["waveform_lengths"].tolist())
            for batch_idx, feature_length in enumerate(feature_lengths_cpu):
                utterance_features = features_cpu[batch_idx, :feature_length].numpy().astype(np.float32)
                units = kmeans.predict(utterance_features).astype(int).tolist()
                sequences.append(units)

    summary = summarize_sequences(sequences, waveform_lengths, int(data_cfg.get("sample_rate", 16000)), args.k)
    summary.update(
        {
            "config": args.config,
            "split": args.split,
            "ssl_model_name": model_cfg["ssl_model_name"],
            "hidden_layer": hidden_layer,
            "k": args.k,
            "max_frames": args.max_frames,
            "inertia": float(kmeans.inertia_),
        }
    )
    save_json(Path(args.output), summary)
    print(summary)


if __name__ == "__main__":
    main()
