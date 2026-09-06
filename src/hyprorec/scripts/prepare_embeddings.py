"""Create offline item representations with lightweight pretrained encoders."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoTokenizer,
    MPNetModel,
    VideoMAEModel,
    ViTModel,
    Wav2Vec2Model,
)

from ..constants import MODALITIES

DEFAULT_MODELS = {
    "txt": "sentence-transformers/all-mpnet-base-v2",
    "img": "google/vit-base-patch16-224-in21k",
    "ado": "facebook/wav2vec2-base",
    "vdo": "MCG-NJU/videomae-base",
}
DEFAULT_BATCH_SIZES = {"txt": 64, "img": 32, "ado": 8, "vdo": 2}
MM_FILE_PREFIX = {"img": "image", "ado": "audio", "vdo": "video"}


def load_texts(dataset_dir: Path) -> list[str]:
    with (dataset_dir / "movies_info.csv").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))
    return [
        f"{row['movieName'].strip()} {row.get('description', '').strip()}".strip()
        for row in rows
    ]


def _block_id(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def iter_multimodal_batches(
    dataset_dir: Path,
    modality: str,
    batch_size: int,
) -> Iterator[np.ndarray]:
    paths = sorted(
        (dataset_dir / "mm").glob(f"{MM_FILE_PREFIX[modality]}_*.npy"),
        key=_block_id,
    )
    if not paths:
        raise FileNotFoundError(f"No raw {modality} files found.")
    for path in paths:
        block = np.load(path, mmap_mode="r")
        for start in range(0, len(block), batch_size):
            yield np.array(block[start : start + batch_size], copy=True)


def _mean_pool(hidden_state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden_state.dtype)
    return (hidden_state * weights).sum(1) / weights.sum(1).clamp_min(1)


def encode_text(
    dataset_dir: Path,
    model_path: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = MPNetModel.from_pretrained(model_path).eval().to(device)
    texts = load_texts(dataset_dir)
    outputs = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="txt embeddings"):
            inputs = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            features = _mean_pool(
                model(**inputs).last_hidden_state, inputs["attention_mask"]
            )
            outputs.append(F.normalize(features, dim=-1).cpu())
    return torch.cat(outputs)


def encode_image(
    dataset_dir: Path,
    model_path: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = ViTModel.from_pretrained(model_path).eval().to(device)
    outputs = []
    with torch.inference_mode():
        for batch in tqdm(
            iter_multimodal_batches(dataset_dir, "img", batch_size),
            desc="img embeddings",
        ):
            inputs = processor(images=list(batch), return_tensors="pt").to(device)
            outputs.append(
                F.normalize(model(**inputs).last_hidden_state[:, 0], dim=-1).cpu()
            )
    return torch.cat(outputs)


def encode_audio(
    dataset_dir: Path,
    model_path: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    processor = AutoFeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path).eval().to(device)
    outputs = []
    with torch.inference_mode():
        for batch in tqdm(
            iter_multimodal_batches(dataset_dir, "ado", batch_size),
            desc="ado embeddings",
        ):
            waveforms = [waveform.astype(np.float32) / 128.0 for waveform in batch]
            inputs = processor(
                waveforms,
                sampling_rate=16000,
                padding=True,
                return_tensors="pt",
            ).to(device)
            features = model(**inputs).last_hidden_state.mean(dim=1)
            outputs.append(F.normalize(features, dim=-1).cpu())
    return torch.cat(outputs)


def encode_video(
    dataset_dir: Path,
    model_path: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = VideoMAEModel.from_pretrained(model_path).eval().to(device)
    outputs = []
    with torch.inference_mode():
        for batch in tqdm(
            iter_multimodal_batches(dataset_dir, "vdo", batch_size),
            desc="vdo embeddings",
        ):
            inputs = processor(
                [list(video) for video in batch], return_tensors="pt"
            ).to(device)
            features = model(**inputs).last_hidden_state.mean(dim=1)
            outputs.append(F.normalize(features, dim=-1).cpu())
    return torch.cat(outputs)


def prepare_embeddings(
    dataset_dir: Path,
    modalities: list[str],
    model_paths: dict[str, str],
    batch_size: int | None,
    device: torch.device,
    output_dir: Path | None = None,
) -> None:
    # START: Preserve native encoder widths and allow raw data and outputs to differ.
    output_dir = output_dir or dataset_dir / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    num_items = len(load_texts(dataset_dir))
    metadata: dict[str, object] = {
        "num_items": num_items,
        "modalities": {},
    }
    encoders = {
        "txt": encode_text,
        "img": encode_image,
        "ado": encode_audio,
        "vdo": encode_video,
    }
    for modality in modalities:
        embeddings = encoders[modality](
            dataset_dir,
            model_paths[modality],
            batch_size or DEFAULT_BATCH_SIZES[modality],
            device,
        )
        if embeddings.ndim != 2 or embeddings.size(0) != num_items:
            raise ValueError(
                f"{modality} embeddings have shape {tuple(embeddings.shape)}, "
                f"expected ({num_items}, feature_dim)."
            )
        torch.save(embeddings.float(), output_dir / f"{modality}_embeddings.pt")
        metadata["modalities"][modality] = {
            "model_name_or_path": model_paths[modality],
            "shape": list(embeddings.shape),
            "normalized": True,
        }
        del embeddings
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output_dir / "embedding_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # END: Preserve native encoder widths and allow raw data and outputs to differ.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    # Modified: write new embeddings without overwriting another project's tables.
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--modality", nargs="+", choices=MODALITIES, default=list(MODALITIES)
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    for modality in MODALITIES:
        parser.add_argument(f"--{modality}-model", default=DEFAULT_MODELS[modality])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_embeddings(
        dataset_dir=args.dataset_dir,
        modalities=args.modality,
        model_paths={
            modality: getattr(args, f"{modality}_model") for modality in MODALITIES
        },
        batch_size=args.batch_size,
        device=torch.device(args.device),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
