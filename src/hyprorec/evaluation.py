"""Metrics used by the native Transformers evaluation loops."""

from __future__ import annotations

from transformers.trainer_utils import EvalPrediction

from .metrics import (
    conversation_teacher_forcing_metrics_from_texts,
    recommendation_metrics,
)


def build_compute_metrics(tokenizer):
    """Build metrics for recommendation and teacher-forced response logits."""

    def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        recommendations, conversation_predictions = prediction.predictions
        conversation_labels, targets = prediction.label_ids
        metrics = recommendation_metrics(
            recommendations.tolist(), targets.tolist()
        )
        references = []
        hypotheses = []
        response_tokens = 0
        for predicted, labels in zip(conversation_predictions, conversation_labels):
            shifted_labels = labels[1:]
            mask = shifted_labels != -100
            response_tokens += int(mask.sum())
            references.append(
                tokenizer.decode(
                    shifted_labels[mask].tolist(), skip_special_tokens=True
                )
            )
            hypotheses.append(
                tokenizer.decode(
                    predicted[mask].tolist(), skip_special_tokens=True
                )
            )
        metrics.update(
            conversation_teacher_forcing_metrics_from_texts(references, hypotheses)
        )
        metrics["response_tokens"] = float(response_tokens)
        return metrics

    return compute_metrics


__all__ = ["build_compute_metrics"]
