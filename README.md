# HyProRec: separate recommendation and conversation tasks

This branch trains recommendation and conversation as independent models and checkpoints. Each input is a task-specific 20-token soft prompt, up to 256 history tokens, and the selected local hypergraph tokens. The four semantic graph views can be combined with one co-occurrence view. The latter uses one selected modality's node features.

Recommendation pools the LM's input-side hidden states and uses either a projected semantic item table (txt, img, ado, vdo, or self-attention fusion of all four) or an MLP classifier. Conversation trains causal language-model loss on every Seeker and Recommender turn. Recommendation trains on item-bearing turns from both roles.

The graph preparation command writes ten candidates per anchor for each semantic view and a co-occurrence table built only from train conversations:

```bash
python -m hyprorec.scripts.prepare_hyperedge_table --dataset-dir data/hocrs2_redial --topk 10
```

Training can select the first `topk` neighbors or randomly sample `topk` from the saved ten candidates. The latter is used only for training; evaluation uses the deterministic first `topk`. `sample_repeat` repeats each training example with a new graph sample.

On Lab, initialize the user's environment before running W&B jobs:

```bash
source ~/lhf/init.sh
python -m hyprorec.scripts.grounding --config configs/redial/hocrs/grounding.yaml
torchrun --standalone --nproc-per-node=8 -m hyprorec.scripts.train --config configs/redial/hocrs/full-item-txt.yaml
```

Grounding trains eight independent graph towers: four modality-similarity towers and four modality-initialized co-occurrence towers. The CRS model loads the four semantic towers and one co-occurrence tower selected by `co_feature_view`. Run Grounding before training a CRS configuration that points to its checkpoint.
