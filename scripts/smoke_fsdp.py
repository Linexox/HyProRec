"""Real FSDP/HSDP train, checkpoint restore and final evaluation."""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.utils.data import Subset
from transformers import TrainerCallback, set_seed

from hyprorec.backbones import load_backbone_adapter
from hyprorec.config import parse_experiment_args
from hyprorec.data import HoCRSDataCollator, HoCRSDataset
from hyprorec.data.hypergraph import HypergraphTable
from hyprorec.model_factory import build_model, load_modality_tables
from hyprorec.processing_hocrs import HoCRSProcessor
from hyprorec.scripts.train import _dataset_config
from hyprorec.training import HyProRecTrainer


class Progress(TrainerCallback):
    def __init__(self):
        self.steps = []

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        row = {'optimizer_step': state.global_step,
               'seconds': time.perf_counter() - self.started}
        self.steps.append(row)
        if args.process_index == 0:
            print(json.dumps(row), flush=True)


class ProbeTrainer(HyProRecTrainer):
    micro_steps = 0
    max_sequence_length = 0

    def training_step(self, model, inputs, *args, **kwargs):
        self.micro_steps += 1
        length = inputs['input_ids'].size(-1)
        self.max_sequence_length = max(self.max_sequence_length, length)
        row = {'rank': self.args.process_index, 'micro_step': self.micro_steps,
               'sequence_length': length,
               'supervised_tokens': int(inputs['labels'].ne(-100).sum())}
        try:
            result = super().training_step(model, inputs, *args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            print(json.dumps(dict(row, oom=True,
                                  allocated_gib=torch.cuda.memory_allocated() / 2**30,
                                  reserved_gib=torch.cuda.memory_reserved() / 2**30)), flush=True)
            raise
        if self.micro_steps % 8 == 0:
            print(json.dumps(dict(row,
                                  allocated_gib=torch.cuda.memory_allocated() / 2**30,
                                  reserved_gib=torch.cuda.memory_reserved() / 2**30)), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    model_args, data_args, args, _ = parse_experiment_args(['--config', str(cli.config)])
    _ = args.device
    set_seed(args.seed)
    adapter = load_backbone_adapter(model_args.backbone_name_or_path)
    processor = HoCRSProcessor(adapter.tokenizer, model_args.num_prompt_tokens)
    tables = load_modality_tables(data_args, model_args)
    model = build_model(model_args, data_args, processor, tables, adapter)
    table = HypergraphTable.from_json(data_args.hyperedge_table_path)
    train = HoCRSDataset(_dataset_config(data_args, data_args.hyperedge_table_path, training=True), 'train', table)
    valid = HoCRSDataset(_dataset_config(data_args, data_args.hyperedge_table_path, training=False), 'validation', table)
    args.output_dir = str(cli.output)
    args.max_steps = 3
    args.eval_steps = args.save_steps = 1
    args.logging_steps = 1
    args.warmup_steps = 0
    args.warmup_ratio = 0.
    args.disable_tqdm = True
    args.report_to = []
    progress = Progress()
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(42))
    count = args.world_size * (2 * args.gradient_accumulation_steps + 5)
    trainer = ProbeTrainer(
        model=model, args=args, processing_class=processor,
        train_dataset=Subset(train, indices[:count].tolist()), callbacks=[progress],
        eval_dataset=Subset(valid, range(17)),
        data_collator=HoCRSDataCollator(processor, data_args.max_history_tokens,
                                      data_args.max_sequence_tokens, views=tuple(data_args.views)),
    )
    trained = trainer.train()
    plugin = trainer.accelerator.state.fsdp_plugin
    assert plugin.fsdp_version == 2 and isinstance(model, FSDPModule)
    mesh = trainer.accelerator.parallelism_config
    assert mesh.dp_shard_size == args.dp_shard_size
    assert mesh.dp_replicate_size == args.dp_replicate_size
    assert args.world_size == args.dp_shard_size * args.dp_replicate_size
    parameter = next(p for p in model.parameters() if p.requires_grad and p.numel())
    assert isinstance(parameter, DTensor)
    local = parameter.to_local()
    expected = local.detach().clone()
    with torch.no_grad():
        local.add_(1)
    trainer._load_best_model()
    assert torch.equal(parameter.to_local().detach(), expected)
    metrics = trainer.evaluate()
    assert torch.isfinite(torch.tensor(metrics['eval_loss']))
    for view, reference in tables.items():
        for prefix in ('', 'item_'):
            assert torch.equal(getattr(model, f'{prefix}{view}_feature_table').cpu(), reference)
    ranks = [None] * args.world_size
    torch.distributed.all_gather_object(ranks, {
        'rank': args.process_index, 'passed': True,
        'peak_memory_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'max_sequence_length': trainer.max_sequence_length,
        'steps': progress.steps,
    })
    if trainer.is_world_process_zero():
        (cli.output / 'smoke.json').write_text(json.dumps({
            'passed': True, 'fsdp_version': plugin.fsdp_version,
            'dp_replicate_size': args.dp_replicate_size,
            'dp_shard_size': args.dp_shard_size,
            'sharded_parameters': True, 'checkpoint_restore_passed': True,
            'feature_tables_unchanged': True, 'ranks': ranks,
            'best_model_checkpoint': trainer.state.best_model_checkpoint,
            'train': trained.metrics, 'eval': metrics,
        }, indent=2))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
