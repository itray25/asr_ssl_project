from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from src.data.vocab import CharVocabulary


class ASRCollator:
    def __init__(self, vocab: CharVocabulary):
        self.vocab = vocab

    def __call__(self, items: list[dict]) -> dict:
        waveforms = []
        waveform_lengths = []
        transcripts = []
        ids = []
        encoded_labels = []

        for item in items:
            waveform = item["waveform"]
            if waveform.dim() == 2:
                waveform = waveform.squeeze(0) if waveform.size(0) == 1 else waveform.mean(dim=0)
            waveform = waveform.float()
            transcript = item["transcript"]
            label = torch.tensor(self.vocab.encode(transcript), dtype=torch.long)

            waveforms.append(waveform)
            waveform_lengths.append(waveform.numel())
            transcripts.append(transcript)
            ids.append(item["id"])
            encoded_labels.append(label)

        padded_waveforms = pad_sequence(waveforms, batch_first=True)
        label_lengths = torch.tensor([label.numel() for label in encoded_labels], dtype=torch.long)
        labels = torch.cat(encoded_labels) if encoded_labels else torch.empty(0, dtype=torch.long)

        return {
            "waveforms": padded_waveforms,
            "waveform_lengths": torch.tensor(waveform_lengths, dtype=torch.long),
            "labels": labels,
            "label_lengths": label_lengths,
            "transcripts": transcripts,
            "ids": ids,
        }
