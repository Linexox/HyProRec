# Backbone inventory

The refactor uses `backbone_name_or_path` and reuses the configured Hugging Face cache; model files are not copied into the repository.

| Backbone | Planned role | Current LAB status |
| --- | --- | --- |
| `Qwen/Qwen2.5-Omni-3B` | primary four-view backbone | available at `/data2/lhf/.MODEL/Qwen2.5-Omni-3B` |
| `sensenova/InteractiveOmni-4B` | four-view comparison | not present in the inspected cache |
| `llava-hf/llava-1.5-7b-hf` | text+image comparison | available in the HF cache |
| `Qwen/Qwen2.5-VL-3B-Instruct` | text+image comparison | not present in the inspected cache |
| `Qwen/Qwen2.5-3B-Instruct` | text-only comparison | not present in the inspected cache |

The missing three weights are left pending. The LAB worktree contains symlinks for the two available checkpoints so experiments can refer to stable worktree-local paths without duplicating files.
