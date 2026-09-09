"""Regression tests for HoCRS2 Grounding and Trainer loss contracts."""

import tempfile
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from transformers import GPT2Config, MPNetConfig, Trainer, TrainingArguments, set_seed

from hyprorec.configuration_grounding import HoCRSGroundingConfig
from hyprorec.configuration_hocrs import HoCRSConfig
from hyprorec.data.grounding import (
    GroundingSourceDataset,
    HoCRSGroundingCollator,
    HoCRSGroundingDataset,
)
from hyprorec.data.hypergraph import HypergraphTable
from hyprorec.modeling_grounding import HoCRSGroundingModel, build_source_config
from hyprorec.modeling_hocrs import HoCRSModel
from hyprorec.arguments import ModelArguments, DataArguments, HoCRSTrainingArguments
from hyprorec.scripts import grounding as grounding_script
from hyprorec.scripts.train import _build_model
from test_modeling_hocrs import build_processor
from transformers import GPT2LMHeadModel
from safetensors.torch import load_file


# START: Verify original source architectures, raw input ordering, and Trainer contracts.
class HoCRSFidelityTest(unittest.TestCase):
    def test_random_item_width_is_independent_of_content_width(self):
        processor = build_processor()
        backbone = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(processor.tokenizer),
                n_embd=2048,
                n_layer=0,
                n_head=16,
                n_positions=16,
                bos_token_id=1,
                eos_token_id=1,
            )
        )
        with mock.patch("hyprorec.scripts.train.load_backbone", return_value=backbone):
            model = _build_model(
                ModelArguments(),
                DataArguments(views=[]),
                processor,
                {},
                torch.randn(3, 256),
            )
        self.assertEqual(
            tuple(model.recommendation_head.user_projector[0].weight.shape),
            (2048, 2048),
        )
        self.assertEqual(
            tuple(model.recommendation_head.user_projector[3].weight.shape),
            (2048, 2048),
        )
        self.assertEqual(
            tuple(model.recommendation_head.item_table.weight.shape), (3, 2048)
        )
        self.assertTrue(model.recommendation_head.item_table.weight.requires_grad)

    def test_sequential_grounding_exports_each_validation_best(self):
        from transformers import ViTConfig

        def tiny_source(view):
            common = dict(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
            )
            return (
                MPNetConfig(**common, vocab_size=20, max_position_embeddings=32)
                if view == "txt"
                else ViTConfig(**common, image_size=16, patch_size=8)
            )

        class Tokenizer:
            def save_pretrained(self, path):
                pass

            def __len__(self):
                return 20

            def __call__(self, values, **kwargs):
                return {"input_ids": torch.tensor([[2, int(x) + 4, 3] for x in values])}

        evaluations = []
        original_evaluate = Trainer.evaluate

        def evaluate(trainer, *args, **kwargs):
            metrics = original_evaluate(trainer, *args, **kwargs)
            self.assertIn("eval_loss", metrics)
            evaluations.append((trainer.model.config.views, trainer.state.epoch))
            # Force epoch 1 to win; verify export does not accidentally use epoch 2.
            metrics["eval_loss"] = float(trainer.state.epoch)
            return metrics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "embeddings").mkdir()
            (root / "mm").mkdir()
            (root / "movies_info.csv").write_text(
                "movieName,description\n" + "\n".join(f"{i},ignored" for i in range(10))
            )
            np.save(
                root / "mm/image_0.npy",
                np.random.default_rng(0).integers(
                    0, 256, (10, 3, 16, 16), dtype=np.uint8
                ),
            )
            rows = [[i, (i + 1) % 10] for i in range(10)]
            (root / "hyperedge_table.json").write_text(
                json.dumps({"txt": rows, "img": rows})
            )
            for view in ["txt", "img"]:
                torch.save(
                    torch.randn(10, 6), root / "embeddings" / f"{view}_embeddings.pt"
                )
            recipe = root / "recipe.yaml"
            recipe.write_text("model: {}\ndata: {}\ntraining: {}\n")
            args = HoCRSTrainingArguments(
                output_dir=str(root / "output"),
                use_cpu=True,
                do_train=True,
                do_eval=True,
                num_train_epochs=2,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                report_to=[],
                per_device_train_batch_size=3,
                per_device_eval_batch_size=1,
                save_total_limit=2,
                disable_tqdm=True,
            )
            parsed = (
                ModelArguments(
                    hypergraph_hidden_dim=8,
                    hypergraph_output_dim=8,
                    hypergraph_num_layers=1,
                ),
                DataArguments(
                    dataset_path=str(root), views=["txt", "img"], topk=1, khop=1
                ),
                args,
                recipe,
            )
            with (
                mock.patch.object(
                    grounding_script, "parse_experiment_args", return_value=parsed
                ),
                mock.patch.object(
                    grounding_script.AutoTokenizer,
                    "from_pretrained",
                    return_value=Tokenizer(),
                ),
                mock.patch.object(
                    grounding_script, "build_source_config", side_effect=tiny_source
                ),
                mock.patch(
                    "hyprorec.modeling_grounding.build_source_config",
                    side_effect=tiny_source,
                ),
                mock.patch.object(Trainer, "evaluate", evaluate),
            ):
                grounding_script.main()
            selections = json.loads((root / "output/best_modalities.json").read_text())
            combined = load_file(str(root / "output/model.safetensors"))
            self.assertEqual(
                [view for view, epoch in evaluations],
                [("txt",), ("txt",), ("img",), ("img",)],
            )
            for view, record in selections.items():
                self.assertEqual(record["eval_loss"], 1.0)
                self.assertTrue(record["checkpoint"].endswith("checkpoint-3"))
                selected = load_file(
                    str(Path(record["checkpoint"]) / "model.safetensors")
                )
                for name, tensor in selected.items():
                    torch.testing.assert_close(combined[name], tensor, rtol=0, atol=0)
            restored = HoCRSGroundingModel.from_pretrained(root / "output")
            for name, tensor in restored.state_dict().items():
                torch.testing.assert_close(combined[name], tensor, rtol=0, atol=0)

    def test_source_architectures_match_hocrs2(self):
        for view, kind, layers in [
            ("txt", "mpnet", 4),
            ("img", "vit", 4),
            ("ado", "wav2vec2", 2),
            ("vdo", "videomae", 4),
        ]:
            config = build_source_config(view)
            self.assertEqual(config.model_type, kind)
            self.assertEqual(config.hidden_size, 256)
            self.assertEqual(config.num_hidden_layers, layers)
            self.assertEqual(config.num_attention_heads, 4)
            self.assertEqual(config.intermediate_size, 1024)
        self.assertEqual(build_source_config("ado").conv_stride, [5, 4, 4, 4])
        self.assertEqual(build_source_config("vdo").num_frames, 16)

    def test_all_original_source_encoders_accept_raw_shapes(self):
        for view in ["txt", "img", "ado", "vdo"]:
            model = HoCRSGroundingModel(
                HoCRSGroundingConfig(views=[view], input_dims={view: 768})
            )
            model.eval()
            raw = {
                "txt": {"input_ids": torch.tensor([[2, 7, 3]])},
                "img": {"pixel_values": torch.rand(1, 3, 224, 224)},
                "ado": {"input_values": torch.rand(1, 16000)},
                "vdo": {"pixel_values": torch.rand(1, 16, 3, 224, 224)},
            }[view]
            output = model.encode_source(view, raw)
            self.assertEqual(tuple(output.shape), (1, 256))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(
                any(
                    p.grad is not None for p in model.source_encoders[view].parameters()
                )
            )
            with tempfile.TemporaryDirectory() as directory:
                model.save_pretrained(directory)
                restored = HoCRSGroundingModel.from_pretrained(directory)
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(
                        tensor, restored.state_dict()[name], rtol=0, atol=0
                    )
                del restored
            del model, output

    def test_raw_collation_preserves_graph_node_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mm").mkdir()
            (root / "movies_info.csv").write_text(
                "movieName,description\nA,ignored\nB,ignored\n"
            )
            np.save(
                root / "mm/audio_0.npy",
                np.array([[64, 32], [128, 0]], dtype=np.float32),
            )
            source = GroundingSourceDataset(root, ["ado"])
            table = HypergraphTable({"ado": [[0, 1], [1, 0]]})
            dataset = HoCRSGroundingDataset(table, ["ado"], topk=1, khop=1)
            features = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
            batch = HoCRSGroundingCollator({"ado": features}, ["ado"], source)(
                [dataset[1]]
            )
            self.assertEqual(batch["hypergraphs"]["ado"]["node_ids"].tolist(), [1, 0])
            torch.testing.assert_close(
                batch["source_data"]["ado"]["input_values"],
                torch.tensor([[1.0, 0.0], [0.5, 0.25]]),
            )
            torch.testing.assert_close(batch["node_features"]["ado"], features[[1, 0]])
            self.assertNotIn("labels", batch)
            source._blocks.clear()  # Modified: release mmap handles before Windows fixture cleanup.

    def test_grounding_raw_forward_backward_eval_and_roundtrip(self):
        set_seed(42)
        source_config = MPNetConfig(
            vocab_size=20,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
        )
        config = HoCRSGroundingConfig(
            views=["txt"],
            input_dims={"txt": 6},
            hidden_dim=8,
            output_dim=8,
            num_layers=1,
            source_configs={"txt": source_config.to_dict()},
        )
        model = HoCRSGroundingModel(config)
        table = HypergraphTable({"txt": [[0, 1], [1, 0], [2, 1]]})
        dataset = HoCRSGroundingDataset(table, ["txt"], topk=1, khop=1)
        tokenizer = lambda values, **kwargs: {
            "input_ids": torch.tensor([[2, int(x) + 4, 3] for x in values]),
            "attention_mask": torch.ones(len(values), 3, dtype=torch.long),
        }
        raw = [{"txt": str(i)} for i in range(3)]
        collator = HoCRSGroundingCollator(
            {"txt": torch.randn(3, 6)}, ["txt"], raw, tokenizer
        )
        batch = collator([dataset[0], dataset[2]])
        output = model(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(
            any(p.grad is not None for p in model.source_encoders.parameters())
        )
        self.assertTrue(
            any(p.grad is not None for p in model.hypergraph_encoders.parameters())
        )
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(
                output_dir=directory,
                use_cpu=True,
                report_to=[],
                label_names=[],
                prediction_loss_only=True,
                per_device_eval_batch_size=2,
            )
            trainer = Trainer(
                model=model, args=args, eval_dataset=dataset, data_collator=collator
            )
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            self.assertIn("eval_loss", trainer.evaluate())
            model.save_pretrained(directory)
            restored = HoCRSGroundingModel.from_pretrained(directory)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(
                    tensor, restored.state_dict()[name], rtol=0, atol=0
                )

    def test_crs_accumulation_matches_large_batch(self):
        def train(batch_size, accumulation):
            set_seed(17)
            config = HoCRSConfig(
                backbone_config=GPT2Config(
                    vocab_size=12,
                    n_embd=8,
                    n_head=2,
                    n_layer=1,
                    n_positions=16,
                    resid_pdrop=0.0,
                    embd_pdrop=0.0,
                    attn_pdrop=0.0,
                ),
                views=[],
                num_items=3,
                item_dim=8,
                recommendation_hidden_dim=8,
                rec_token_id=3,
                num_soft_prompt_tokens=0,
                freeze_backbone=True,
            )
            model = HoCRSModel(config)
            samples = [
                {
                    "input_ids": torch.tensor([2, 3, 4, 5]),
                    "attention_mask": torch.ones(4, dtype=torch.long),
                    "labels": torch.tensor([-100, -100, 4, 5]),
                    "rec_labels": torch.tensor(i),
                }
                for i in [0, 1]
            ]
            with tempfile.TemporaryDirectory() as directory:
                args = TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                    report_to=[],
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=accumulation,
                    max_steps=1,
                    max_grad_norm=0.0,
                    save_strategy="no",
                    learning_rate=0.1,
                    label_names=["labels", "rec_labels"],
                )
                optimizer = torch.optim.SGD(
                    [p for p in model.parameters() if p.requires_grad], lr=0.1
                )
                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=samples,
                    optimizers=(
                        optimizer,
                        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
                    ),
                )
                self.assertFalse(trainer.model_accepts_loss_kwargs)
                batch = {
                    name: torch.stack([sample[name] for sample in samples])
                    for name in samples[0]
                }
                expected_loss = model(**batch).loss.detach()
                with mock.patch.object(
                    type(trainer.accelerator),
                    "num_processes",
                    new_callable=mock.PropertyMock,
                    return_value=8,
                ):
                    actual_loss = trainer.compute_loss(
                        model, dict(batch), num_items_in_batch=torch.tensor(32)
                    )
                torch.testing.assert_close(actual_loss.detach(), expected_loss)
                trainer.train()
            return model.recommendation_head.state_dict()

        large, accumulated = train(2, 1), train(1, 2)
        for key in large:
            torch.testing.assert_close(
                large[key], accumulated[key], rtol=1e-4, atol=1e-5
            )


# END: Verify original source architectures, raw input ordering, and Trainer contracts.


if __name__ == "__main__":
    unittest.main()
