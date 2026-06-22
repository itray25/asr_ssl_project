from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.text import normalize_transcript


@dataclass(frozen=True)
class _IndexedItem:
    index: int
    duration: float | None = None


class LibriSpeechDataset(Dataset):
    def __init__(self, config: dict, split: str, train: bool):
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for LibriSpeech experiments.") from exc

        self.torchaudio = torchaudio
        data_cfg = config["data"]
        self.target_sample_rate = int(data_cfg.get("sample_rate", 16000))
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=str(Path(data_cfg.get("root", "data")) / "LibriSpeech" / split),
            url=split,
            download=bool(data_cfg.get("download", False)),
        )
        self.indices = self._select_indices(data_cfg, train)

    def _select_indices(self, data_cfg: dict, train: bool) -> list[_IndexedItem]:
        max_samples = data_cfg.get("max_train_samples") if train else data_cfg.get("max_eval_samples")
        max_samples = int(max_samples) if max_samples is not None else None

        if train and data_cfg.get("max_train_hours") is not None:
            max_seconds = float(data_cfg["max_train_hours"]) * 3600.0
            selected = []
            total_seconds = 0.0
            for idx in range(len(self.dataset)):
                waveform, sample_rate, *_ = self.dataset[idx]
                duration = float(waveform.size(-1)) / float(sample_rate)
                if total_seconds >= max_seconds:
                    break
                selected.append(_IndexedItem(idx, duration))
                total_seconds += duration
                if max_samples is not None and len(selected) >= max_samples:
                    break
            return selected

        limit = len(self.dataset) if max_samples is None else min(max_samples, len(self.dataset))
        return [_IndexedItem(idx) for idx in range(limit)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        indexed = self.indices[item]
        waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id = self.dataset[indexed.index]
        if sample_rate != self.target_sample_rate:
            waveform = self.torchaudio.functional.resample(waveform, sample_rate, self.target_sample_rate)
            sample_rate = self.target_sample_rate
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0) if waveform.size(0) == 1 else waveform.mean(dim=0)
        return {
            "id": f"{speaker_id}-{chapter_id}-{utterance_id}",
            "waveform": waveform.to(torch.float32),
            "sample_rate": int(sample_rate),
            "transcript": normalize_transcript(transcript),
        }


def build_dataset(config: dict, split: str, train: bool) -> LibriSpeechDataset:
    data_cfg = config.get("data", {})
    if data_cfg.get("dataset", "librispeech") != "librispeech":
        raise ValueError(f"Unsupported dataset: {data_cfg.get('dataset')}")
    return LibriSpeechDataset(config, split, train)
