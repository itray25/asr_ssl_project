from __future__ import annotations

from pathlib import Path
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.collate import ASRCollator
from src.data.librispeech import build_dataset
from src.data.vocab import CharVocabulary
from src.models.logmel_ctc import LogMelCTC
from src.models.ssl_ctc import SSLCTC
from src.training.decoding import greedy_ctc_decode
from src.training.metrics import asr_metrics
from src.training.utils import append_log, copy_config, resolve_device, save_json, set_seed


def build_model(config: dict, vocab_size: int) -> nn.Module:
    model_cfg = config["model"]
    data_cfg = config["data"]
    model_type = model_cfg["type"]
    if model_type == "logmel_ctc":
        return LogMelCTC(
            vocab_size=vocab_size,
            n_mels=int(model_cfg.get("n_mels", 80)),
            hidden_size=int(model_cfg.get("hidden_size", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            sample_rate=int(data_cfg.get("sample_rate", 16000)),
        )
    if model_type == "ssl_ctc":
        return SSLCTC(
            vocab_size=vocab_size,
            ssl_model_name=model_cfg["ssl_model_name"],
            freeze_encoder=bool(model_cfg.get("freeze_encoder", True)),
            hidden_layer=model_cfg.get("hidden_layer"),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def build_loaders(config: dict, vocab: CharVocabulary):
    training_cfg = config["training"]
    data_cfg = config["data"]
    train_dataset = build_dataset(config, data_cfg.get("train_split", "train-clean-100"), train=True)
    dev_dataset = build_dataset(config, data_cfg.get("dev_split", "dev-clean"), train=False)
    collator = ASRCollator(vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        collate_fn=collator,
    )
    return train_loader, dev_loader


def compute_ctc_loss(outputs: dict, batch: dict, blank_id: int) -> torch.Tensor:
    loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    return loss_fn(
        outputs["log_probs"],
        batch["labels"],
        outputs["output_lengths"].cpu(),
        batch["label_lengths"].cpu(),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    vocab: CharVocabulary,
    device: torch.device,
    predictions_path: str | Path | None = None,
) -> dict:
    model.eval()
    references = []
    hypotheses = []
    rows = []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["waveforms"], batch["waveform_lengths"])
        decoded = greedy_ctc_decode(outputs["log_probs"], outputs["output_lengths"], vocab)
        references.extend(batch["transcripts"])
        hypotheses.extend(decoded)
        rows.extend(
            {
                "id": item_id,
                "reference": ref,
                "hypothesis": hyp,
            }
            for item_id, ref, hyp in zip(batch["ids"], batch["transcripts"], decoded)
        )

    metrics = asr_metrics(references, hypotheses)
    if predictions_path is not None:
        predictions_path = Path(predictions_path)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def train_experiment(config: dict, config_path: str | Path) -> dict:
    exp_cfg = config["experiment"]
    training_cfg = config["training"]
    output_dir = Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    set_seed(int(exp_cfg.get("seed", 1337)))
    device = resolve_device(training_cfg.get("device", "auto"))
    vocab = CharVocabulary.default()
    vocab.save(output_dir / "vocab.json")
    copy_config(config_path, output_dir)

    train_loader, dev_loader = build_loaders(config, vocab)
    model = build_model(config, vocab.size).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    best_wer = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history = []
    append_log(log_path, f"device={device}")

    for epoch in range(1, int(training_cfg.get("epochs", 1)) + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["waveforms"], batch["waveform_lengths"])
            loss = compute_ctc_loss(outputs, batch, vocab.blank_id)
            loss.backward()
            grad_clip = training_cfg.get("grad_clip")
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            epoch_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(len(train_loader), 1)
        dev_metrics = evaluate_model(
            model,
            dev_loader,
            vocab,
            device,
            predictions_path=output_dir / "dev_predictions.jsonl",
        )
        elapsed = time.time() - start
        record = {"epoch": epoch, "train_loss": train_loss, "dev": dev_metrics, "seconds": elapsed}
        history.append(record)
        append_log(log_path, json.dumps(record, ensure_ascii=False))

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "vocab": vocab.token_to_id,
            "epoch": epoch,
            "dev_metrics": dev_metrics,
        }
        torch.save(checkpoint, last_path)
        if dev_metrics["wer"] <= best_wer:
            best_wer = dev_metrics["wer"]
            torch.save(checkpoint, best_path)

    save_json(output_dir / "metrics.json", {"history": history, "best_dev_wer": best_wer})
    return {"output_dir": str(output_dir), "best_checkpoint": str(best_path), "history": history}


def evaluate_checkpoint(config: dict, checkpoint_path: str | Path, split: str) -> dict:
    device = resolve_device(config["training"].get("device", "auto"))
    output_dir = Path(config["experiment"]["output_dir"])
    vocab_path = output_dir / "vocab.json"
    vocab = CharVocabulary.load(vocab_path) if vocab_path.exists() else CharVocabulary.default()

    model = build_model(config, vocab.size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = build_dataset(config, split, train=False)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=ASRCollator(vocab),
    )
    split_name = split.replace("/", "_")
    metrics = evaluate_model(
        model,
        loader,
        vocab,
        device,
        predictions_path=output_dir / f"{split_name}_predictions.jsonl",
    )
    save_json(output_dir / f"{split_name}_metrics.json", metrics)
    return metrics

