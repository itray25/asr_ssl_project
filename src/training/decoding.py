from __future__ import annotations

import torch


def greedy_ctc_decode(log_probs: torch.Tensor, output_lengths: torch.Tensor, vocab) -> list[str]:
    """Decode T x B x V log probabilities with CTC collapse."""
    pred_ids = log_probs.argmax(dim=-1).transpose(0, 1).cpu()
    hypotheses = []
    for seq, length in zip(pred_ids, output_lengths.cpu()):
        collapsed = []
        previous = None
        for token_id in seq[: int(length.item())].tolist():
            if token_id != previous and token_id != vocab.blank_id:
                collapsed.append(token_id)
            previous = token_id
        hypotheses.append(vocab.decode(collapsed))
    return hypotheses

