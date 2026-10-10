# Backbone inventory

The refactor uses `backbone_name_or_path` and reuses the configured Hugging Face cache; model files are not copied into the repository.

| Backbone | Planned role | Current LAB status |
| --- | --- | --- |
| `Qwen/Qwen2.5-Omni-3B` | primary four-view backbone | available at `/data2/lhf/.MODEL/Qwen2.5-Omni-3B` |
| `Qwen/Qwen2.5-Omni-7B` | existing large-backbone smoke/config | available in the HF cache and linked under `backbones/` |
| `sensenova/InteractiveOmni-4B` | four-view comparison | available in the LAB HF cache and HoCRS-smoke-tested |
| `llava-hf/llava-1.5-7b-hf` | text+image comparison | available in the HF cache |
| `Qwen/Qwen2.5-VL-3B-Instruct` | text+image comparison | available in the LAB HF cache and smoke-tested |
| `Qwen/Qwen2.5-3B-Instruct` | text-only comparison | download in progress in the LAB HF cache |

The LAB worktree contains symlinks for the available checkpoints so experiments can refer to stable worktree-local paths without duplicating files. The Qwen text checkpoint is still being resumed in place and will be linked only after its snapshot contains every shard.
