"""Six-rank graph/text CRS regression before TG-ReDial eta/LR calibration."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import set_seed

from hyprorec.backbones import load_backbone_adapter
from hyprorec.config import parse_experiment_args
from hyprorec.data import HoCRSDataCollator, HoCRSDataset
from hyprorec.data.hypergraph import HypergraphTable
from hyprorec.model_factory import build_model, load_modality_tables
from hyprorec.processing_hocrs import HoCRSProcessor
from hyprorec.scripts.train import _dataset_config
from hyprorec.training import HyProRecTrainer


class CalibrationSmokeTrainer(HyProRecTrainer):
    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        if output.num_samples != 17 or self._diagnostic_count['loss/fusion'] != 17:
            raise ValueError('Validation padding was not removed from component metrics.')
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/tg_redial/full-calibration-eta1-lr5e-4-e12.yaml'))
    cli = parser.parse_args()
    model_args, data_args, args, _ = parse_experiment_args([
        '--config', str(cli.config),
    ])
    _ = args.device
    if args.world_size != 6:
        raise ValueError('Calibration smoke requires six ranks.')
    effective_batch = args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps
    if effective_batch != 258 or args.per_device_train_batch_size != 1:
        raise ValueError('Calibration smoke requires per-device batch 1 and effective batch 258.')
    set_seed(args.seed)
    adapter = load_backbone_adapter(model_args.backbone_name_or_path)
    processor = HoCRSProcessor(adapter.tokenizer, model_args.num_prompt_tokens)
    tables = load_modality_tables(data_args, model_args)
    model = build_model(model_args, data_args, processor, tables, adapter)
    table = HypergraphTable.from_json(data_args.hyperedge_table_path)
    train = HoCRSDataset(_dataset_config(data_args, data_args.hyperedge_table_path, training=True), 'train', table)
    valid = HoCRSDataset(_dataset_config(data_args, data_args.hyperedge_table_path, training=False), 'validation', table)
    args.output_dir = str(cli.output)
    args.logging_dir = str(cli.output / 'trainer-logs')
    args.max_steps = 2
    args.save_strategy = 'steps'
    args.save_steps = 1
    args.load_best_model_at_end = True
    args.save_total_limit = 2
    args.eval_steps = 1
    args.warmup_steps = 0
    args.warmup_ratio = 0.0
    args.logging_steps = 20
    args.disable_tqdm = True
    args.report_to = []
    args.label_names = ['labels', 'rec_labels']
    trainer = CalibrationSmokeTrainer(
        model=model, args=args, processing_class=processor,
        # One full accumulation window and one partial window on every rank.
        train_dataset=Subset(train, range(args.world_size * (args.gradient_accumulation_steps + 5))),
        eval_dataset=Subset(valid, range(17)),
        data_collator=HoCRSDataCollator(processor, data_args.max_history_tokens,
                                      data_args.max_sequence_tokens, views=tuple(data_args.views)),
    )
    trained = trainer.train()
    if trainer.state.global_step != 2 or trainer.state.best_model_checkpoint is None:
        raise ValueError('Training/checkpoint selection did not finish.')
    # Verify actual checkpoint restoration, rather than checkpoint presence alone.
    parameter = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    restored = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1)
    trainer._load_best_model()
    if not torch.equal(parameter.detach(), restored):
        raise ValueError('Best checkpoint restoration changed model weights.')
    model.eval()
    probe = trainer._prepare_inputs(trainer.data_collator([valid[16]]))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        injections = [model._inject(probe['input_ids'], probe['hypergraphs']) for _ in range(3)]
        ordinary_delta = max((injections[0] - value).abs().max().item() for value in injections[1:])
        torch.use_deterministic_algorithms(True)
        injections = [model._inject(probe['input_ids'], probe['hypergraphs']) for _ in range(3)]
        deterministic_delta = max((injections[0] - value).abs().max().item() for value in injections[1:])
    first, second = trainer.evaluate(), trainer.evaluate()
    for row in [r for r in trainer.state.log_history if 'eval_loss' in r] + [first, second]:
        weighted = (row['eval_loss/fusion']
                    + model_args.recommendation_view_loss_weight * row['eval_loss/view']
                    + model_args.conversation_loss_weight * row['eval_loss/conversation'])
        if abs(row['eval_loss'] - weighted) > 1e-4 or row['eval_loss/total'] != row['eval_loss']:
            raise ValueError(f'Loss reductions disagree: {row}')
    if abs(first['eval_loss'] - second['eval_loss']) > 1e-5:
        raise ValueError('Repeated fixed-model validation changed.')
    if not torch.isfinite(torch.tensor(trained.metrics['train_loss'])):
        raise ValueError('Nonfinite training loss.')
    ranks = [None] * args.world_size
    torch.distributed.all_gather_object(ranks, {'rank': args.process_index, 'passed': True})
    if trainer.is_world_process_zero():
        (cli.output / 'smoke.json').write_text(json.dumps({
            'passed': True, 'world_size': args.world_size, 'effective_batch_size': effective_batch,
            'checkpoint_restore_passed': True,
            'best_model_checkpoint': trainer.state.best_model_checkpoint,
            'lambda': model_args.conversation_loss_weight,
            'eta': model_args.recommendation_view_loss_weight,
            'ranks': ranks, 'validation_samples': 17,
            'injection_delta_default': ordinary_delta,
            'injection_delta_deterministic': deterministic_delta,
            'repeat_eval_deterministic_algorithms': True,
            'first': first, 'second': second, 'train': trained.metrics,
        }, indent=2))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
