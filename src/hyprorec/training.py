"""Trainer extensions for unified recommendation and dialogue training."""

from __future__ import annotations

import math

import torch
from transformers import Trainer


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu())
    return float(value)


class HyProRecTrainer(Trainer):
    """Add model diagnostics and unified recommendation/dialogue predictions."""

    def __init__(self, *args, attention_token_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_token_ids = attention_token_ids or {}
        self._diagnostic_sum = {}
        self._diagnostic_count = {}

    def _record(self, key, value, prefix=""):
        value = _as_float(value)
        if value is None or not math.isfinite(value):
            return
        name = f"{prefix}{key}" if prefix else key
        self._diagnostic_sum[name] = self._diagnostic_sum.get(name, 0.0) + value
        self._diagnostic_count[name] = self._diagnostic_count.get(name, 0) + 1

    @staticmethod
    def _unwrapped(model):
        return getattr(model, "module", model)

    def _attention_diagnostics(self, inputs, attention_weights, prefix):
        input_ids = inputs.get("input_ids")
        pooling_mask = inputs.get("pooling_mask")
        if input_ids is None or pooling_mask is None or not attention_weights:
            return
        token_ids = self.attention_token_ids
        soft_id = token_ids.get("soft_prompt_token_id")
        starts = token_ids.get("graph_start_token_ids", {})
        ends = token_ids.get("graph_end_token_ids", {})
        node_id = token_ids.get("node_token_id")
        hyperedge_id = token_ids.get("hyperedge_token_id")
        if soft_id is None:
            return

        input_ids = input_ids.detach()
        valid = pooling_mask.bool()
        soft_mask = input_ids.eq(soft_id) & valid
        source_masks = {}
        all_graph = torch.zeros_like(valid)
        for view, start_id in starts.items():
            end_id = ends.get(view)
            if end_id is None:
                continue
            section = torch.zeros_like(valid)
            for row in range(input_ids.size(0)):
                positions = torch.nonzero(input_ids[row].eq(start_id), as_tuple=False)
                if positions.numel() == 0:
                    continue
                start = int(positions[0].item())
                end_positions = torch.nonzero(
                    input_ids[row, start + 1 :].eq(end_id), as_tuple=False
                )
                if end_positions.numel() == 0:
                    continue
                end = start + 1 + int(end_positions[0].item())
                section[row, start + 1 : end] = True
            section &= valid
            if node_id is None or hyperedge_id is None:
                mask = section
            else:
                mask = section & (input_ids.eq(node_id) | input_ids.eq(hyperedge_id))
            source_masks[view] = mask
            all_graph |= section

        context_mask = valid & ~soft_mask & ~all_graph
        for view, weights in attention_weights.items():
            weights = weights.float()
            self._record(
                f"attention/{view}/soft_prompt_mass",
                (weights * soft_mask.float()).sum(dim=-1).mean(),
                prefix,
            )
            self._record(
                f"attention/{view}/context_mass",
                (weights * context_mask.float()).sum(dim=-1).mean(),
                prefix,
            )
            self._record(
                f"attention/{view}/hypergraph_mass",
                (weights * all_graph.float()).sum(dim=-1).mean(),
                prefix,
            )
            for source_view, source_mask in source_masks.items():
                self._record(
                    f"attention/{view}/{source_view}_hypergraph_mass",
                    (weights * source_mask.float()).sum(dim=-1).mean(),
                    prefix,
                )
            clipped = weights.clamp_min(1e-8)
            self._record(
                f"attention/{view}/entropy",
                -(clipped * clipped.log()).sum(dim=-1).mean(),
                prefix,
            )
            self._record(
                f"attention/{view}/top5_mass",
                weights.topk(min(5, weights.size(-1)), dim=-1).values.sum(dim=-1).mean(),
                prefix,
            )
            self._record(
                f"attention/{view}/max_weight",
                weights.max(dim=-1).values.mean(),
                prefix,
            )

    def _collect_diagnostics(self, outputs, inputs, model):
        # ``eval_*`` is rewritten by Transformers' W&B callback.  The
        # underscore prefix keeps diagnostics under ``eval/`` instead of
        # ``train/eval/...``.
        prefix = "" if model.training else "eval_"
        for name in (
            "loss",
            "fusion_loss",
            "view_loss",
            "bpr_loss",
            "balance_loss",
            "conversation_loss",
        ):
            field = "total" if name == "loss" else name.removesuffix("_loss")
            self._record(f"loss/{field}", getattr(outputs, name, None), prefix)

        route = getattr(outputs, "route_weights", None)
        if route is not None:
            route = route.float()
            means = route.mean(dim=0)
            head = self._unwrapped(model).recommendation_head
            for view, value in zip(head.names, means):
                self._record(f"rec_router/mean_{view}", value, prefix)
            clipped = route.clamp_min(1e-8)
            self._record(
                "rec_router/entropy",
                -(clipped * clipped.log()).sum(dim=-1).mean(),
                prefix,
            )
            self._record(
                "rec_router/top1_fraction",
                route.argmax(dim=-1).eq(means.argmax()).float().mean(),
                prefix,
            )
            self._record("rec_router/std", route.std(unbiased=False), prefix)

        for view, values in (getattr(outputs, "graph_moe_diagnostics", None) or {}).items():
            for index, value in enumerate(values.get("expert_mean", ())):
                self._record(f"graph_moe/{view}/expert_{index}_mean", value, prefix)
            self._record(f"graph_moe/{view}/entropy", values.get("entropy"), prefix)
            for index, value in enumerate(values.get("top1_fraction", ())):
                self._record(
                    f"graph_moe/{view}/expert_{index}_top1_fraction", value, prefix
                )
            top1_fraction = values.get("top1_fraction")
            if top1_fraction is not None:
                self._record(
                    f"graph_moe/{view}/top1_fraction", top1_fraction.max(), prefix
                )
            self._record(
                f"graph_moe/{view}/dead_expert_fraction",
                values.get("dead_expert_fraction"),
                prefix,
            )

        self._attention_diagnostics(
            inputs, getattr(outputs, "attention_weights", None), prefix
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with self.compute_loss_context_manager():
            outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        self._collect_diagnostics(outputs, inputs, model)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs
    ):
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach().mean()
        if prediction_loss_only:
            return loss, None, None
        recommendations = outputs.recommendation_logits.topk(
            min(50, outputs.recommendation_logits.size(-1)), dim=-1
        ).indices
        conversation_predictions = outputs.conversation_logits[:, :-1].argmax(dim=-1)
        labels = (inputs["labels"].detach(), inputs["rec_labels"].detach())
        return loss, (recommendations.detach(), conversation_predictions.detach()), labels

    def log(self, logs, *args, **kwargs):
        logs = dict(logs)
        if "learning_rate" in logs:
            logs["optimization/learning_rate"] = logs["learning_rate"]
        if "grad_norm" in logs:
            logs["optimization/grad_norm"] = logs["grad_norm"]
        for key, total in self._diagnostic_sum.items():
            logs[key] = total / max(1, self._diagnostic_count[key])
        self._diagnostic_sum.clear()
        self._diagnostic_count.clear()
        return super().log(logs, *args, **kwargs)


__all__ = ["HyProRecTrainer"]
