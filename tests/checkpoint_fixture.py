"""Reproduce the rank-zero checkpoint publication race with real Trainer I/O."""
import json
import sys
import time
from pathlib import Path
import torch
from transformers import Trainer, TrainingArguments
from hyprorec.training import StageAwareTrainerMixin


class DelayedCheckpointTrainer(StageAwareTrainerMixin, Trainer):
    def _save(self, output_dir=None, state_dict=None):
        time.sleep(.5)
        return super()._save(output_dir, state_dict)


root = Path(sys.argv[1])
args = TrainingArguments(output_dir=str(root), use_cpu=True, ddp_backend='gloo',
    save_strategy='steps', save_only_model=True, report_to=[], disable_tqdm=True)
trainer = DelayedCheckpointTrainer(model=torch.nn.Linear(2, 2), args=args)
trainer.state.global_step = 1
trainer.state.best_global_step = 1
trainer.state.best_metric = 1.0
trainer._save_checkpoint(trainer.model, None)
rank = torch.distributed.get_rank()
(root / f'rank-{rank}.json').write_text(json.dumps({'best':trainer.state.best_model_checkpoint}))
torch.distributed.barrier()
torch.distributed.destroy_process_group()
