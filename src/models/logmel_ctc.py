from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogMelCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 80,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = 400
        self.win_length = 400
        self.hop_length = 160

        self.mel_transform = self._build_mel_transform()
        self.encoder = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def _build_mel_transform(self):
        try:
            import torchaudio

            return torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                center=False,
                power=2.0,
            )
        except Exception:
            return None

    def _feature_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        lengths = ((waveform_lengths - self.win_length) // self.hop_length) + 1
        return lengths.clamp_min(1)

    def _fallback_log_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win_length, device=waveforms.device)
        spec = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        spec = spec.pow(2).clamp_min(1e-10).log()
        spec = F.interpolate(spec.unsqueeze(1), size=(self.n_mels, spec.size(-1)), mode="bilinear")
        return spec.squeeze(1).transpose(1, 2)

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        if self.mel_transform is not None:
            self.mel_transform = self.mel_transform.to(waveforms.device)
            feats = self.mel_transform(waveforms).clamp_min(1e-10).log().transpose(1, 2)
        else:
            feats = self._fallback_log_features(waveforms)

        mean = feats.mean(dim=1, keepdim=True)
        std = feats.std(dim=1, keepdim=True).clamp_min(1e-5)
        return (feats - mean) / std

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> dict:
        features = self.extract_features(waveforms)
        feature_lengths = self._feature_lengths(waveform_lengths).to(waveforms.device)
        feature_lengths = feature_lengths.clamp_max(features.size(1))

        packed = nn.utils.rnn.pack_padded_sequence(
            features,
            feature_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.encoder(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(self.dropout(encoded))
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return {"log_probs": log_probs, "output_lengths": feature_lengths}

