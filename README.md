# HyProRec

HyProRec is the clean implementation of the first end-to-end HoCRS model. It
uses one dialogue co-occurrence hypergraph and up to four modality-similarity
hypergraphs (`txt`, `img`, `ado`, and `vdo`) as prompts for an optional causal
language-model backbone. Grounding and TASK are intentionally absent from v1.

## Method boundary

- Co-occurrence edges are built from unique items mentioned in one **training
  dialogue**. Validation and test dialogues never contribute to topology.
- Each semantic view is built only from nearest neighbors in that modality.
- Every view has an independent HGCN and graph-to-LM projector. Graph tokens
  from different views are not aligned or fused position by position.
- The co-occurrence node table and the trainable recommendation item table are
  separate tensors. The Item Table is initialized from the normalized content
  basis and optimized independently. Co-occurrence node features are an
  independent trainable table with random initialization unless an explicit
  `co` table is supplied.
- With semantic views enabled, the content basis is the normalized mean of
  those enabled modalities. A co-only run falls back to all four modalities.
- An empty `views` list is a valid no-hypergraph baseline: no topology file is
  loaded and no HGCN/projector is constructed. Its Item Table still receives
  the fused content initialization from all four modalities.

## Data layout

```text
data/lhf-redial/
├── train_data.json
├── valid_data.json
├── test_data.json
├── movies_info.csv
├── hyperedge_table.json
├── embeddings/
│   ├── txt_embeddings.pt
│   ├── img_embeddings.pt
│   ├── ado_embeddings.pt
│   └── vdo_embeddings.pt
└── mm/
    └── ... raw modality blocks used only by embedding preparation
```

Prepare offline features and the separated hyperedge table:

```bash
hyprorec-prepare-embeddings --dataset-dir data/lhf-redial
hyprorec-prepare-hyperedges --dataset-dir data/lhf-redial --topk 50
```

`hyperedge_table.json` has five explicit top-level keys: `co`, `txt`, `img`,
`ado`, and `vdo`. Each row starts with its anchor item ID and is followed by a
ranked neighbor list.

## Training

The entry point reads one sectioned YAML file. Any leaf can still be overridden
with a normal Transformers-style command-line argument.

```bash
hyprorec-train --config configs/redial/v1.yaml

# Example override
hyprorec-train --config configs/redial/v1.yaml \
  --per_device_train_batch_size 2 --run_name smoke

# No-hypergraph ablation: use a recipe whose `data.views` is `[]`
hyprorec-train --config configs/redial/v1/no_hypergraph.yaml
```

The default recipe selects `best` through Transformers' standard
`load_best_model_at_end` logic using the minimum `eval_loss`. Test data is not
read unless `do_predict: true`. Checkpoints contain the complete HoCRS state,
including all active HGCNs, projectors, feature tables, the item table, soft
prompts, trainable special-token deltas, and the LM backbone.
Validation logs include total loss, recommendation loss, and conversation loss.

Set `WANDB_PROJECT` in `.env` when `report_to: wandb` is enabled.

## Distributed launch

Ordinary DDP needs no model-specific switch:

```bash
torchrun --nproc-per-node=4 -m hyprorec.scripts.train \
  --config configs/redial/v1.yaml
```

FSDP2 uses a one-dimensional sharding mesh:

```bash
torchrun --nproc-per-node=4 -m hyprorec.scripts.train \
  --config configs/redial/v1.yaml \
  --dp_shard_size 4
```

HSDP uses replication groups over sharded groups. The product must equal the
world size:

```bash
torchrun --nproc-per-node=4 -m hyprorec.scripts.train \
  --config configs/redial/v1.yaml \
  --dp_replicate_size 2 --dp_shard_size 2
```

Saved models use the normal `save_pretrained` layout and can be reloaded after
importing `hyprorec`:

```python
from hyprorec import HoCRSModel, HoCRSProcessor

processor = HoCRSProcessor.from_pretrained("outputs/redial/v1/full")
model = HoCRSModel.from_pretrained("outputs/redial/v1/full")
```
