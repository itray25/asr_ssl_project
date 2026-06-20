from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLCTC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ssl_model_name: str,
        freeze_encoder: bool = True,
        hidden_layer: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for wav2vec2/HuBERT experiments.") from exc

        self.encoder = AutoModel.from_pretrained(ssl_model_name)
        self.hidden_layer = hidden_layer
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def _output_lengths(self, waveform_lengths: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "_get_feat_extract_output_lengths"):
            return self.encoder._get_feat_extract_output_lengths(waveform_lengths).long()
        return torch.div(waveform_lengths, 320, rounding_mode="floor").clamp_min(1)

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> dict:
        attention_mask = torch.arange(waveforms.size(1), device=waveforms.device).unsqueeze(0)
        attention_mask = attention_mask < waveform_lengths.to(waveforms.device).unsqueeze(1)

        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.encoder(
                waveforms,
                attention_mask=attention_mask.long(),
                output_hidden_states=self.hidden_layer is not None,
            )

        if self.hidden_layer is None:
            features = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            if self.hidden_layer < 0 or self.hidden_layer >= len(hidden_states):
                raise ValueError(
                    f"hidden_layer={self.hidden_layer} is out of range for "
                    f"{len(hidden_states)} hidden-state tensors."
                )
            features = hidden_states[self.hidden_layer]

        logits = self.classifier(self.dropout(features))
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        output_lengths = self._output_lengths(waveform_lengths).to(waveforms.device)
        output_lengths = output_lengths.clamp_max(features.size(1))
        return {"log_probs": log_probs, "output_lengths": output_lengths}

