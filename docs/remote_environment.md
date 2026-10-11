# LAB refactor environment

The remote worktree is `/data2/lhf/HyProRec/refactor`. Its project-specific
environment is `/data2/lhf/envs/hyprorec-refactor`, created with the existing
qwen-omni environment as read-only system packages. It contains Torch
`2.6.0+cu118`, Transformers `4.53.3`, Accelerate `1.14.0`, SwanLab `0.10.2`,
and the data/evaluation dependencies used by the repository.

The InteractiveOmni custom-code smoke additionally uses `decord`, `onnxruntime`,
`timm`, `librosa`, `torchaudio 2.6.0+cu118`, and `diffusers 0.34.0`. These are
installed only in this new project environment; existing LAB environments are
unchanged.

Launch commands should use:

```bash
source ~/lhf/init.sh
export HYPROREC_PYTHON=/data2/lhf/envs/hyprorec-refactor/bin/python
export PYTHONPATH=/data2/lhf/HyProRec/refactor/src
```

The environment does not contain an API key. Put `SWANLAB_API_KEY` and, when
needed, `SWANLAB_API_HOST` in a worktree-local `.env` before online training;
`.env` is ignored by git. `SWANLAB_MODE=disabled` is the supported smoke-test
mode. The init script's legacy W&B login command is not used by refactor code.

The current four-view ReDial Grounding smoke artifact is kept outside the
repository at `/data2/lhf/tmp/hyprorec-grounding-smoke`; it contains the merged
`model.safetensors`, per-view checkpoints, `best_modalities.json`, and the
resolved recipe. It was produced with batch size 8 and one optimizer step using
the feature-projection fallback for ReDial's absent raw image, video, and audio
blocks. A Qwen2.5-Omni-3B CRS single-sample restore smoke loads this artifact
and returns recommendation logits over all 6,924 items plus teacher-forced
dialogue predictions.
