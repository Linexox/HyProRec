"""Trainer extensions for unified recommendation and dialogue training."""

from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


def canonical_stage_logs(stage, logs):
    prefix = f"{stage}_"
    return {
        key.removeprefix(prefix): value
        for key, value in logs.items()
    }


class StageAwareTrainerMixin:
    def __init__(self, *args, **kwargs):
        self._metric_stage = "train"
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.args.world_size > 1:
            # The writer owns checkpoint publication. Filesystem visibility
            # must not make ranks take different best-model restore paths.
            selected = [self.state.best_model_checkpoint]
            torch.distributed.broadcast_object_list(
                selected, src=0, device=self.args.device
            )
            self.state.best_model_checkpoint = selected[0]

    @contextmanager
    def metric_stage(self, stage):
        previous = self._metric_stage
        self._metric_stage = stage
        try:
            yield
        finally:
            self._metric_stage = previous

    def evaluate(self, *args, **kwargs):
        with self.metric_stage("eval"):
            return super().evaluate(*args, **kwargs)


def validation_steps(updates_per_epoch: int, total_steps: int) -> tuple[int, ...]:
    """Return midpoint and end optimizer steps for each training epoch."""

    if updates_per_epoch < 1 or total_steps < 1:
        return ()

    steps = []
    for epoch_start in range(0, total_steps, updates_per_epoch):
        epoch_steps = min(updates_per_epoch, total_steps - epoch_start)
        for fraction in (0.5, 1.0):
            step = epoch_start + math.ceil(epoch_steps * fraction)
            if step not in steps:
                steps.append(step)
    return tuple(steps)


class TwoValidChecksCallback(TrainerCallback):
    """Request evaluation and checkpointing at the two scheduled steps."""

    def __init__(self, steps: tuple[int, ...]):
        self.steps = frozenset(steps)

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step in self.steps:
            control.should_evaluate = True
            control.should_save = True
        return control


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().float().cpu())
    return float(value)


class HyProRecTrainer(StageAwareTrainerMixin, Trainer):
    """Add model diagnostics and unified recommendation/dialogue predictions."""

    def __init__(self, *args, attention_token_ids=None, swanlab_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_token_ids = attention_token_ids or {}
        self.swanlab_tracker = swanlab_tracker
        self._diagnostic_sum = {}
        self._diagnostic_count = {}

    @contextmanager
    def metric_stage(self, stage):
        previous = self._diagnostic_sum, self._diagnostic_count
        self._diagnostic_sum, self._diagnostic_count = {}, {}
        try:
            with super().metric_stage(stage):
                yield
        finally:
            self._diagnostic_sum, self._diagnostic_count = previous

    def _record(self, key, value, weight=1):
        value = _as_float(value)
        if value is None or not math.isfinite(value):
            return
        self._diagnostic_sum[key] = self._diagnostic_sum.get(key, 0.0) + value * weight
        self._diagnostic_count[key] = self._diagnostic_count.get(key, 0) + weight

    @staticmethod
    def _unwrapped(model):
        return getattr(model, "module", model)

    def _attention_diagnostics(self, inputs, attention_weights):
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
            )
            self._record(
                f"attention/{view}/context_mass",
                (weights * context_mask.float()).sum(dim=-1).mean(),
            )
            self._record(
                f"attention/{view}/hypergraph_mass",
                (weights * all_graph.float()).sum(dim=-1).mean(),
            )
            for source_view, source_mask in source_masks.items():
                self._record(
                    f"attention/{view}/{source_view}_hypergraph_mass",
                    (weights * source_mask.float()).sum(dim=-1).mean(),
                )

    def _collect_diagnostics(self, outputs, inputs, model):
        if model.training:
            for name in ("loss", "fusion_loss", "view_loss", "conversation_loss"):
                field = "total" if name == "loss" else name.removesuffix("_loss")
                self._record(f"loss/{field}", getattr(outputs, name, None))

        # Attention and routing summaries scan token masks and are useful for
        # diagnosis, but they do not affect optimization.  Collect them only
        # during evaluation so gradient accumulation does not repeat that work
        # for every training micro-step.
        collect_auxiliary = not model.training
        route = getattr(outputs, "route_weights", None)
        if route is not None and collect_auxiliary:
            route = route.float()
            means = route.mean(dim=0)
            head = self._unwrapped(model).recommendation_head
            for view, value in zip(head.names, means):
                self._record(f"rec_router/mean_{view}", value)

        if collect_auxiliary:
            self._attention_diagnostics(
                inputs, getattr(outputs, "attention_weights", None)
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
        fields = {
            name: getattr(outputs, name, None)
            for name in ("fusion_loss", "view_loss", "conversation_loss")
            if getattr(outputs, name, None) is not None
        }
        if fields:
            # Match Trainer's sample-weighted loss reduction, including removal
            # of the examples Accelerate repeats to pad the final DDP batch.
            values = torch.stack([value.detach().float() for value in fields.values()])
            values = values.unsqueeze(0).expand(inputs["input_ids"].size(0), -1)
            values = self.accelerator.gather_for_metrics(values)
            for name, value in zip(fields, values.mean(dim=0)):
                self._record(f"loss/{name.removesuffix('_loss')}", value, values.size(0))
        if prediction_loss_only:
            return loss, None, None
        recommendations = outputs.recommendation_logits.topk(
            min(50, outputs.recommendation_logits.size(-1)), dim=-1
        ).indices
        conversation_predictions = outputs.conversation_predictions
        labels = (inputs["labels"].detach(), inputs["rec_labels"].detach())
        return loss, (recommendations.detach(), conversation_predictions.detach()), labels

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
        output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        for key, total in self._diagnostic_sum.items():
            if key.startswith("loss/"):
                output.metrics[f"{metric_key_prefix}_{key}"] = total / self._diagnostic_count[key]
        native_loss = output.metrics.get(f"{metric_key_prefix}_loss")
        if native_loss is not None:
            output.metrics[f"{metric_key_prefix}_loss/total"] = native_loss
        return output

    def predict(self, *args, **kwargs):
        with self.metric_stage("test"):
            output = super().predict(*args, **kwargs)
            self.log(output.metrics, stage="test")
        return output

    def log(self, logs, *args, stage=None, **kwargs):
        logs = dict(logs)
        metric_stage = stage or self._metric_stage
        if "learning_rate" in logs:
            logs["optimization/learning_rate"] = logs["learning_rate"]
        if "grad_norm" in logs:
            logs["optimization/grad_norm"] = logs["grad_norm"]
        if metric_stage == "train" and self.args.world_size > 1:
            names = [f"loss/{name}" for name in ("total", "fusion", "view", "conversation")]
            statistics = torch.tensor(
                [[self._diagnostic_sum.get(key, 0.0), self._diagnostic_count.get(key, 0)]
                 for key in names], device=self.args.device, dtype=torch.float64,
            )
            statistics = self.accelerator.reduce(statistics, reduction="sum").tolist()
            for key, (total, count) in zip(names, statistics):
                if count:
                    self._diagnostic_sum[key], self._diagnostic_count[key] = total, count
        for key, total in self._diagnostic_sum.items():
            name = key if metric_stage == "train" else f"{metric_stage}_{key}"
            logs.setdefault(name, total / max(1, self._diagnostic_count[key]))
        self._diagnostic_sum.clear()
        self._diagnostic_count.clear()
        if self.swanlab_tracker is not None and self.is_world_process_zero():
            self.swanlab_tracker.log_trainer(
                metric_stage,
                canonical_stage_logs(metric_stage, logs),
            )
        return super().log(logs, *args, **kwargs)


__all__ = [
    "TwoValidChecksCallback",
    "HyProRecTrainer",
    "StageAwareTrainerMixin",
    "canonical_stage_logs",
    "validation_steps",
]
