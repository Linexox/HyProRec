import tempfile
import unittest
from unittest import mock

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from hyprorec.configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from hyprorec.data import HoCRSDataCollator
from hyprorec.data.hypergraph import HypergraphData
from hyprorec.modeling_hocrs import HoCRSModel, MOE_NUM_EXPERTS
from hyprorec.processing_hocrs import HoCRSProcessor


def build_processor() -> HoCRSProcessor:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[EOS]": 1,
                "User": 2,
                ":": 3,
                "Assistant": 4,
                "hello": 5,
                "reply": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    return HoCRSProcessor(fast, num_soft_prompt_tokens=2)


def build_model(processor: HoCRSProcessor) -> HoCRSModel:
    backbone_config = GPT2Config(
        vocab_size=len(processor.tokenizer),
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=128,
        eos_token_id=1,
        pad_token_id=1,
    )
    graph_config = HoCRSHypergraphConfig(
        input_dim=8,
        hidden_dim=8,
        output_dim=8,
        num_layers=1,
    )
    config = HoCRSConfig(
        backbone_config=backbone_config,
        views=["txt"],
        txt_hypergraph_config=graph_config,
        num_items=3,
        item_feature_dim=8,
        item_table_view="txt",
        recommendation_hidden_dim=8,
        num_soft_prompt_tokens=2,
        freeze_backbone=True,
        train_special_tokens=True,
        **processor.get_token_id_map(),
    )
    model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
    model.initialize_feature_tables({"txt": torch.randn(3, 8)})
    return model


def build_batch(processor: HoCRSProcessor, graph: bool = True):
    hypergraphs = (
        {"txt": HypergraphData.from_hyperedges("txt", [(0, [1])])} if graph else {}
    )
    return HoCRSDataCollator(processor)(
        [
            {
                "context": "hello",
                "target_item_id": 1,
                "response": "reply",
                "hypergraphs": hypergraphs,
            }
        ]
    )


class HoCRSModelTest(unittest.TestCase):
    def test_collator_masks_only_input_side_tokens_for_preference(self) -> None:
        processor = build_processor()
        batch = build_batch(processor)
        expected = batch["attention_mask"].bool() & batch["labels"].eq(-100)
        rec_token_id = processor.get_token_id_map()["rec_token_id"]
        expected &= batch["input_ids"].ne(rec_token_id)

        torch.testing.assert_close(batch["preference_mask"], expected)
        self.assertGreater(int(batch["preference_mask"].sum()), 0)
        self.assertFalse(bool(batch["preference_mask"][batch["labels"].ne(-100)].any()))

    def test_forward_uses_pooling_and_one_semantic_item_table(self) -> None:
        processor = build_processor()
        model = build_model(processor)
        output = model(**build_batch(processor))

        self.assertEqual(output.rec_scores.shape, (1, 3))
        self.assertFalse(hasattr(model.recommendation_head, "item_table"))
        self.assertEqual(model.config.item_table_view, "txt")
        output.loss.backward()
        self.assertIsNotNone(model.recommendation_head.query_projector.weight.grad)
        self.assertIsNotNone(model.recommendation_head.item_projector.weight.grad)
        self.assertIsNotNone(
            model.hypergraph_projectors["txt"].node_projector.weight.grad
        )

    def test_item_table_view_can_differ_from_graph_views(self) -> None:
        processor = build_processor()
        backbone_config = GPT2Config(
            vocab_size=len(processor.tokenizer),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=128,
            eos_token_id=1,
            pad_token_id=1,
        )
        graph_config = HoCRSHypergraphConfig(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_layers=1,
        )
        config = HoCRSConfig(
            backbone_config=backbone_config,
            views=["txt"],
            txt_hypergraph_config=graph_config,
            num_items=3,
            item_feature_dim=6,
            item_table_view="img",
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=2,
            **processor.get_token_id_map(),
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
        txt = torch.randn(3, 8)
        img = torch.randn(3, 6)
        model.initialize_feature_tables({"txt": txt, "img": img})

        torch.testing.assert_close(model.txt_feature_table, txt)
        torch.testing.assert_close(model.img_feature_table, img)
        self.assertEqual(model(**build_batch(processor)).rec_scores.shape, (1, 3))

    def test_graphless_samples_still_use_semantic_candidates(self) -> None:
        processor = build_processor()
        model = build_model(processor)
        output = model(**build_batch(processor, graph=False))

        self.assertEqual(output.rec_scores.shape, (1, 3))
        self.assertEqual(output.moe_usage.shape, (MOE_NUM_EXPERTS,))
        self.assertTrue(torch.isfinite(output.moe_router_entropy))

    def test_preference_mask_is_required_for_recommendation_loss(self) -> None:
        processor = build_processor()
        model = build_model(processor)
        batch = build_batch(processor)
        batch.pop("preference_mask")

        with self.assertRaisesRegex(ValueError, "preference_mask"):
            model(**batch)

    def test_standard_save_reload_preserves_feature_tables(self) -> None:
        processor = build_processor()
        model = build_model(processor)
        batch = build_batch(processor)
        generation_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "hypergraphs": batch["hypergraphs"],
        }
        projector = model.hypergraph_projectors["txt"]
        with mock.patch.object(projector, "forward", wraps=projector.forward) as call:
            generated = model.generate(
                **generation_inputs,
                min_new_tokens=2,
                max_new_tokens=2,
            )
        self.assertEqual(generated.shape[1], batch["input_ids"].shape[1] + 2)
        self.assertEqual(call.call_count, 1)

        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            model.save_pretrained(directory)
            restored = HoCRSModel.from_pretrained(directory)
            auto_model = AutoModelForCausalLM.from_pretrained(directory)

        self.assertIsInstance(auto_model, HoCRSModel)
        torch.testing.assert_close(model.txt_feature_table, restored.txt_feature_table)
        self.assertEqual(restored.config.item_table_view, "txt")

    def test_wrapping_keeps_pretrained_backbone_weights(self) -> None:
        processor = build_processor()
        backbone_config = GPT2Config(
            vocab_size=len(processor.tokenizer),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=128,
            eos_token_id=1,
            pad_token_id=1,
        )
        backbone = GPT2LMHeadModel(backbone_config)
        with torch.no_grad():
            backbone.transformer.wte.weight.fill_(0.125)
        graph_config = HoCRSHypergraphConfig(
            input_dim=8,
            hidden_dim=8,
            output_dim=8,
            num_layers=1,
        )
        config = HoCRSConfig(
            backbone_config=backbone_config,
            views=["txt"],
            txt_hypergraph_config=graph_config,
            num_items=3,
            item_feature_dim=8,
            item_table_view="txt",
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=0,
            train_special_tokens=False,
        )
        model = HoCRSModel(config, backbone)

        self.assertAlmostEqual(
            float(model.backbone.transformer.wte.weight.mean()),
            0.125,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
