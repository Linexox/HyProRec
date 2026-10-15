"""Exercise real distributed evaluation on five samples across four ranks."""
import json
from pathlib import Path
import sys

import torch

from test_training_metrics import make_trainer


root = Path(sys.argv[1])
trainer, recorder, _ = make_trainer(root)
trainer.model.train()
trainer.compute_loss(trainer.model, {'input_ids': torch.tensor([[torch.distributed.get_rank() + 1]])})
metrics = trainer.evaluate()
trainer.log({'loss': 26.0})
if trainer.is_world_process_zero():
    metrics['train_fusion'] = recorder.records[-1][1]['loss/fusion']
(root / f"rank-{torch.distributed.get_rank()}.json").write_text(json.dumps(metrics))
torch.distributed.barrier()
torch.distributed.destroy_process_group()
