# HyProRec

<!-- START: Document the v3 aligned-source and joint-Grounding pipeline. -->
## v3 pipeline

The `v3` branch separates raw multimodal alignment from CRS training so raw
image, audio, and video arrays are not loaded in every recommendation step.

1. Four trainable pretrained source encoders and modality-specific linear heads
   map catalogue items into one normalized space. Training averages symmetric
   multi-positive InfoNCE over every available modality pair.
2. The best alignment checkpoint exports fixed per-item modality tables.
3. Enabled modality tables are fused with a masked normalized mean. The result
   independently initializes the co-occurrence node table and Item Table.
4. CRS jointly trains each semantic HGCN with anchor-to-source and
   anchor-to-neighborhood Grounding losses. `co` receives CRS supervision only.
5. An optional shared source projector transforms both semantic graph inputs and
   Grounding targets. Its source-to-neighborhood term is disabled when the
   projector is absent.

The CRS objective is

$$
\mathcal{L}=\beta\mathcal{L}_{rec}+(1-\beta)\mathcal{L}_{conv}
+\lambda_g\mathcal{L}_{ground}.
$$

Run the stages in order:

```bash
torchrun --nproc-per-node=4 -m hyprorec.scripts.align \
  --config configs/redial/v3/alignment.yaml

hyprorec-export-aligned \
  --checkpoint outputs/redial/v3/alignment \
  --dataset-dir data/lhf-redial \
  --output-dir data/lhf-redial/embeddings_v3

hyprorec-prepare-content-table \
  --embedding-dir data/lhf-redial/embeddings_v3 \
  --output data/lhf-redial/content_tables_v3/full.pt \
  --modality txt img ado vdo

hyprorec-prepare-hyperedges \
  --dataset-dir data/lhf-redial \
  --embedding-dir data/lhf-redial/embeddings_v3 \
  --output data/lhf-redial/hyperedge_table_v3.json --topk 50

torchrun --nproc-per-node=4 -m hyprorec.scripts.train \
  --config configs/redial/v3/full.yaml
```

The v3 folder also contains explicit recipes for no Grounding, no hypergraph,
direct graph projection, no soft prompt, random Item Table initialization, and
the trainable source projector. The direct-projection and no-hypergraph recipes
set `grounding_weight: 0` because no encoded semantic anchor exists to ground.
<!-- END: Document the v3 aligned-source and joint-Grounding pipeline. -->

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
<!-- START: Document offline fixed-slot content initialization. -->
- The co-occurrence node table and recommendation Item Table are separate
  trainable tensors initialized from the same offline content table.
- Content tables concatenate normalized modality embeddings in fixed
  `txt/img/ado/vdo` slots. Disabled modality slots are zero, so every ablation
  retains the same width and parameter capacity.
- A co-only run uses the full four-modality content table; `co-*` runs use the
  matching single-modality content table.
<!-- END: Document offline fixed-slot content initialization. -->
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
├── embeddings_v1/
│   ├── txt_embeddings.pt
│   ├── img_embeddings.pt
│   ├── ado_embeddings.pt
│   └── vdo_embeddings.pt
├── content_tables_v1/
│   ├── full.pt
│   ├── txt.pt
│   ├── img.pt
│   ├── ado.pt
│   └── vdo.pt
└── mm/
    └── ... raw modality blocks used only by embedding preparation
```

<!-- START: Document separate raw encoding and content-table preparation. -->
Prepare native-width embeddings, fixed-slot content tables, and hyperedges:

```bash
hyprorec-prepare-embeddings --dataset-dir data/lhf-redial \
  --output-dir data/lhf-redial/embeddings_v1
hyprorec-prepare-content-table --embedding-dir data/lhf-redial/embeddings_v1 \
  --output data/lhf-redial/content_tables_v1/full.pt \
  --modality txt img ado vdo
hyprorec-prepare-hyperedges --dataset-dir data/lhf-redial \
  --embedding-dir data/lhf-redial/embeddings_v1 \
  --output data/lhf-redial/hyperedge_table_v1.json --topk 50
```
<!-- END: Document separate raw encoding and content-table preparation. -->

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
