"""Recommendation and teacher-forced response metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import jieba


def _tokens(text: str, metric_tokenization: str) -> list[str]:
    if metric_tokenization == "whitespace":
        return str(text).strip().split()
    if metric_tokenization == "jieba":
        return [
            token for token in jieba.lcut(str(text), cut_all=False, HMM=True)
            if token.strip()
        ]
    raise ValueError("metric_tokenization must be whitespace or jieba.")


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _bleu(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    n: int,
) -> float:
    clipped = [0] * n
    total = [0] * n
    reference_length = 0
    hypothesis_length = 0
    for reference_tokens, hypothesis_tokens in zip(references, hypotheses):
        reference_length += len(reference_tokens)
        hypothesis_length += len(hypothesis_tokens)
        for order in range(1, n + 1):
            reference_counts = Counter(_ngrams(reference_tokens, order))
            hypothesis_counts = Counter(_ngrams(hypothesis_tokens, order))
            clipped[order - 1] += sum(
                min(count, reference_counts[gram])
                for gram, count in hypothesis_counts.items()
            )
            total[order - 1] += sum(hypothesis_counts.values())
    if (
        hypothesis_length == 0
        or any(count == 0 for count in total)
        or any(count == 0 for count in clipped)
    ):
        return 0.0
    brevity_penalty = (
        1.0
        if hypothesis_length > reference_length
        else math.exp(1.0 - reference_length / hypothesis_length)
    )
    return float(
        brevity_penalty
        * math.exp(
            sum(math.log(match / count) for match, count in zip(clipped, total)) / n
        )
    )


def _distinct(hypotheses: Sequence[Sequence[str]], n: int) -> float:
    grams = [gram for tokens in hypotheses for gram in _ngrams(tokens, n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def recommendation_metrics(
    recommendations: Sequence[Sequence[int]],
    targets: Sequence[int],
) -> dict[str, float]:
    ranks = [
        list(ranking).index(target)
        if target in ranking
        else math.inf
        for ranking, target in zip(recommendations, targets)
    ]
    metrics: dict[str, float] = {}
    for k in (1, 10, 50):
        metrics[f"recall@{k}"] = (
            sum(rank < k for rank in ranks) / len(ranks)
            if ranks
            else 0.0
        )
        metrics[f"mrr@{k}"] = (
            sum(1.0 / (rank + 1) for rank in ranks if rank < k) / len(ranks)
            if ranks
            else 0.0
        )
        metrics[f"ndcg@{k}"] = (
            sum(1.0 / math.log2(rank + 2) for rank in ranks if rank < k) / len(ranks)
            if ranks
            else 0.0
        )
    return metrics


def conversation_teacher_forcing_texts(
    logits,
    labels,
    tokenizer,
) -> tuple[list[str], list[str], int]:
    """Decode teacher-forced predictions and references for one batch."""

    token_predictions = logits[:, :-1].argmax(dim=-1).detach().cpu()
    shifted_labels = labels[:, 1:].detach().cpu()
    references = []
    hypotheses = []
    token_count = 0
    for predicted, target in zip(token_predictions, shifted_labels):
        mask = target != -100
        token_count += int(mask.sum())
        reference = tokenizer.decode(target[mask].tolist(), skip_special_tokens=True)
        hypothesis = tokenizer.decode(predicted[mask].tolist(), skip_special_tokens=True)
        references.append(reference)
        hypotheses.append(hypothesis)
    return references, hypotheses, token_count


def conversation_teacher_forcing_metrics(
    logits,
    labels,
    tokenizer,
    metric_tokenization: str = "whitespace",
) -> dict[str, float]:
    """Compute response metrics from teacher-forced next-token logits."""

    references, hypotheses, token_count = conversation_teacher_forcing_texts(
        logits, labels, tokenizer
    )
    metrics = conversation_teacher_forcing_metrics_from_texts(
        references, hypotheses, metric_tokenization
    )
    metrics["response_tokens"] = float(token_count)
    return metrics


def conversation_teacher_forcing_metrics_from_texts(
    references: Sequence[str],
    hypotheses: Sequence[str],
    metric_tokenization: str = "whitespace",
) -> dict[str, float]:
    """Compute word metrics from decoded texts without special tokens."""
    reference_tokens = [_tokens(text, metric_tokenization) for text in references]
    hypothesis_tokens = [_tokens(text, metric_tokenization) for text in hypotheses]
    metrics = {}
    metrics.update({
        f"bleu@{n}": _bleu(reference_tokens, hypothesis_tokens, n)
        for n in range(1, 5)
    })
    metrics.update({
        f"dist@{n}": _distinct(hypothesis_tokens, n)
        for n in range(1, 5)
    })
    return metrics


__all__ = [
    "conversation_teacher_forcing_metrics",
    "conversation_teacher_forcing_metrics_from_texts",
    "conversation_teacher_forcing_texts",
    "recommendation_metrics",
]
