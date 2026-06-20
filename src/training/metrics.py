from __future__ import annotations


def _edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        curr = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    try:
        from jiwer import wer

        return float(wer(references, hypotheses))
    except Exception:
        edits = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.split()
            hyp_words = hyp.split()
            edits += _edit_distance(ref_words, hyp_words)
            total += len(ref_words)
        return edits / max(total, 1)


def corpus_cer(references: list[str], hypotheses: list[str]) -> float:
    try:
        from jiwer import cer

        return float(cer(references, hypotheses))
    except Exception:
        edits = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            edits += _edit_distance(list(ref), list(hyp))
            total += len(ref)
        return edits / max(total, 1)


def asr_metrics(references: list[str], hypotheses: list[str]) -> dict:
    return {
        "wer": corpus_wer(references, hypotheses),
        "cer": corpus_cer(references, hypotheses),
    }

