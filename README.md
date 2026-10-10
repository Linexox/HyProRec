# HyProRec: hypergraph-prompted conversational recommendation

HyProRec trains one end-to-end model for conversational recommendation. Each
recommendation-bearing turn supplies both an item label and its response
label. The model injects a fixed soft prompt and local hypergraph tokens into
the language-model backbone, then optimizes the recommendation objective and
teacher-forced causal-language objective together:

\[
L = L_{\mathrm{fusion}} + \lambda_{\mathrm{view}}L_{\mathrm{view}}
  + \lambda_{\mathrm{conv}}L_{\mathrm{conv}},
\]

The recommendation head pools the backbone states, projects the user
representation separately for the enabled semantic views (txt, img, ado, and
vdo), scores the corresponding item tables, and uses a sample-level router to
fuse the view scores.

The graph preparation command writes ten cosine-similarity candidates per
anchor for each semantic view:

```bash
python -m hyprorec.scripts.prepare_hyperedge_table --dataset-dir data/hocrs2_redial --topk 10
```

Training can select the first `topk` neighbors or randomly sample `topk` from
the saved ten candidates. Random sampling is used only for training;
evaluation uses the deterministic first `topk`. `sample_repeat` repeats each
training example with a new graph sample.

On Lab, initialize the user's environment before running jobs:

```bash
source ~/lhf/init.sh
python -m hyprorec.scripts.grounding \
  --config configs/redial/hocrs/grounding.yaml
torchrun --standalone --nproc-per-node=8 \
  -m hyprorec.scripts.train \
  --config configs/redial/hocrs/full-item-full-attn.yaml
```

Grounding trains four independent graph towers, one per semantic modality. The
CRS model loads these towers before the unified recommendation-and-dialogue
training run.
