# LAB refactor environment

The remote worktree is `/data2/lhf/HyProRec/refactor`. Its project-specific
environment is `/data2/lhf/envs/hyprorec-refactor`, created with the existing
qwen-omni environment as read-only system packages. It contains Torch
`2.6.0+cu118`, Transformers `4.53.3`, Accelerate `1.14.0`, SwanLab `0.10.2`,
and the data/evaluation dependencies used by the repository.

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
