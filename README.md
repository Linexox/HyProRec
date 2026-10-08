# HyProRec: hypergraph-prompted conversational recommendation

HyProRec trains one end-to-end model for conversational recommendation. Each
recommendation-bearing turn supplies both an item label and its response
label. The model injects a fixed soft prompt and local hypergraph tokens into
the language-model backbone, then optimizes the recommendation objective and
teacher-forced causal-language objective together:

\[
L = L_{\mathrm{fusion}} + \lambda_{\mathrm{view}}L_{\mathrm{view}}
  + \lambda_{\mathrm{bpr}}L_{\mathrm{bpr}}
  + \lambda_{\mathrm{bal}}L_{\mathrm{bal}}
  + \lambda_{\mathrm{conv}}L_{\mathrm{conv}},
\]

where the BPR and router-balance terms are disabled unless their configuration
switches are enabled.

The recommendation head pools the backbone states, projects the user
representation separately for the enabled semantic views (txt, img, ado, and
vdo), scores the corresponding item tables, and uses a sample-level router to
fuse the view scores. The four semantic graph views may be combined with the
co-occurrence view. The latter uses the configured modality feature table for
its graph initialization.

The graph preparation command writes ten candidates per anchor for each semantic view and a co-occurrence table built only from train conversations:

```bash
python -m hyprorec.scripts.prepare_hyperedge_table --dataset-dir data/hocrs2_redial --topk 10
```

Training can select the first `topk` neighbors or randomly sample `topk` from
the saved ten candidates. Random sampling is used only for training;
evaluation uses the deterministic first `topk`. `sample_repeat` repeats each
training example with a new graph sample.

On Lab, initialize the user's environment before running W&B jobs:

```bash
source ~/lhf/init.sh
python -m hyprorec.scripts.grounding \
  --config configs/redial/hocrs/grounding.yaml
torchrun --standalone --nproc-per-node=8 \
  -m hyprorec.scripts.train \
  --config configs/redial/hocrs/full-item-full-attn.yaml
```

Grounding trains eight independent graph towers: four modality-similarity
towers and four modality-initialized co-occurrence towers. The CRS model loads
the four semantic towers and the configured co-occurrence tower before the
unified recommendation-and-dialogue training run.
