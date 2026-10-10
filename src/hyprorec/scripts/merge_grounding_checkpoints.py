"""Copy a semantic grounding checkpoint into a new experiment directory.

Grounding now produces one checkpoint containing the four semantic towers. The
module name is retained so old launch scripts fail predictably at the command
line instead of importing the removed checkpoint-merging implementation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_checkpoint(source_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    shutil.copytree(source_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    copy_checkpoint(args.source, args.output)


if __name__ == "__main__":
    main()
