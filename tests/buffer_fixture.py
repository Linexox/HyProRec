"""Check that native DDP preserves each modality's feature tables."""
import os

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import GPT2Config, set_seed

from hyprorec.configuration_hocrs import HoCRSConfig
from hyprorec.modeling_hocrs import HoCRSRecommendationModel


set_seed(42)
rank = int(os.environ['RANK'])
dimensions = {'txt': 4, 'img': 4, 'ado': 2, 'vdo': 2}
graph_configs = {
    f'{view}_hypergraph_config': {
        'input_dim': size, 'hidden_dim': 8, 'output_dim': 4, 'num_layers': 1,
    }
    for view, size in dimensions.items()
}
config = HoCRSConfig(
    backbone_config=GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=32),
    views=list(dimensions), num_items=3, item_feature_dims=dimensions,
    item_table_view='full', recommendation_hidden_dim=16,
    recommendation_attention_dim=4, recommendation_router_hidden_dim=8,
    prompt_token_id=3, **graph_configs,
)
model = HoCRSRecommendationModel(config)
tables = {
    view: torch.arange(3 * size).reshape(3, size).float() + 100 * (index + 1)
    for index, (view, size) in enumerate(dimensions.items())
}
model.initialize_feature_tables(tables)
torch.distributed.init_process_group(
    'gloo', init_method=os.environ['DDP_STORE'], rank=rank, world_size=2,
)
try:
    wrapped = DistributedDataParallel(model)
    observed = {}
    for phase in ('construction', 'forward'):
        if phase == 'forward':
            input_ids = torch.zeros(1, 24, dtype=torch.long)
            input_ids[:, :20] = 3
            wrapped(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        for view, expected in tables.items():
            for prefix in ('', 'item_'):
                actual = getattr(model, f'{prefix}{view}_feature_table')
                observed[phase, prefix, view] = actual.clone()
finally:
    torch.distributed.destroy_process_group()
for (phase, prefix, view), actual in observed.items():
    torch.testing.assert_close(
        actual, tables[view], rtol=0, atol=0,
        msg=f'rank {rank}: {prefix}{view} changed after DDP {phase}',
    )
