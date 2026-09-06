"""Recommendation and teacher-forced response metrics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from transformers import EvalPrediction


def _tokens(text: str) -> list[str]:
    return str(text).strip().split()


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _bleu(references: Sequence[str], hypotheses: Sequence[str], n: int) -> float:
    clipped = [0] * n
    total = [0] * n
    reference_length = 0
    hypothesis_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        reference_tokens = _tokens(reference)
        hypothesis_tokens = _tokens(hypothesis)
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


def _distinct(hypotheses: Sequence[str], n: int) -> float:
    grams = [gram for text in hypotheses for gram in _ngrams(_tokens(text), n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def recommendation_metrics(
    recommendations: Sequence[Sequence[int]],
    targets: Sequence[int],
) -> dict[str, float]:
    ranks = [
        list(ranking).index(target) if target in ranking else math.inf
        for ranking, target in zip(recommendations, targets)
    ]
    metrics: dict[str, float] = {}
    for k in (1, 10, 50):
        metrics[f"recall@{k}"] = (
            sum(rank < k for rank in ranks) / len(ranks) if ranks else 0.0
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


def preprocess_logits_for_metrics(logits, _labels):
    lm_logits, rec_scores, rec_loss, conv_loss = logits
    topk = rec_scores.topk(min(50, rec_scores.size(-1)), dim=-1).indices
    token_predictions = lm_logits[:, :-1].argmax(dim=-1)
    batch_size = rec_scores.size(0)
    return (
        topk,
        token_predictions,
        rec_loss.reshape(1).expand(batch_size),
        conv_loss.reshape(1).expand(batch_size),
    )


def build_compute_metrics(processor) -> callable:
    tokenizer = processor.tokenizer

    def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        recommendations, token_predictions, rec_losses, conv_losses = (
            prediction.predictions
        )
        lm_labels, rec_labels = prediction.label_ids
        shifted_labels = lm_labels[:, 1:]
        references = []
        hypotheses = []
        for predicted, labels in zip(token_predictions, shifted_labels):
            mask = labels != -100
            references.append(tokenizer.decode(labels[mask], skip_special_tokens=True))
            hypotheses.append(
                tokenizer.decode(predicted[mask], skip_special_tokens=True)
            )
        metrics = recommendation_metrics(recommendations.tolist(), rec_labels.tolist())
        metrics["rec_loss"] = float(rec_losses.mean())
        metrics["conv_loss"] = float(conv_losses.mean())
        for n in range(1, 5):
            metrics[f"bleu@{n}"] = _bleu(references, hypotheses, n)
            metrics[f"dist@{n}"] = _distinct(hypotheses, n)
        return metrics

    return compute_metrics


__all__ = [
    "build_compute_metrics",
    "preprocess_logits_for_metrics",
    "recommendation_metrics",
]
