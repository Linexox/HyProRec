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
from hyprorec.data.hypergraph import HypergraphData
from hyprorec.data.redial import HoCRSDataCollator
from hyprorec.modeling_hocrs import HoCRSModel
from hyprorec.processing_hocrs import CONTEXT_TOKEN, HoCRSProcessor


def build_processor(use_context_token: bool = False) -> HoCRSProcessor:
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
    return HoCRSProcessor(
        fast,
        num_soft_prompt_tokens=2,
        use_context_token=use_context_token,
    )


def build_model(processor: HoCRSProcessor) -> HoCRSModel:
    backbone_config = GPT2Config(
        vocab_size=len(processor.tokenizer),
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=128,
        bos_token_id=None,
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
        item_dim=8,
        recommendation_hidden_dim=8,
        num_soft_prompt_tokens=2,
        freeze_backbone=True,
        train_special_tokens=True,
        use_context_token=processor.use_context_token,
        **processor.get_token_id_map(),
    )
    model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
    model.initialize_feature_tables({"txt": torch.randn(3, 8)})
    return model


class HoCRSModelTest(unittest.TestCase):
    def test_context_token_follows_history_in_all_prompts(self) -> None:
        processor = build_processor(use_context_token=True)

        graph_prompt = processor.build_prompt("hello", {"txt": (1, 1)})
        graphless_prompt = processor.build_prompt("hello", {})

        for prompt in (graph_prompt, graphless_prompt):
            self.assertEqual(prompt.count(CONTEXT_TOKEN), 1)
            self.assertGreater(prompt.index(CONTEXT_TOKEN), prompt.index("hello"))
        self.assertLess(graph_prompt.index(CONTEXT_TOKEN), graph_prompt.index("Hypergraphs:"))
        self.assertLess(graphless_prompt.index(CONTEXT_TOKEN), graphless_prompt.index("Recommendation state:"))
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            restored = HoCRSProcessor.from_pretrained(directory)
        self.assertTrue(restored.use_context_token)

    def test_context_residual_projector_starts_neutral_and_receives_gradients(self) -> None:
        processor = build_processor(use_context_token=True)
        model = build_model(processor)
        batch = HoCRSDataCollator(processor)(
            [{
                "context": "hello",
                "target_item_id": 1,
                "response": "reply",
                "hypergraphs": {"txt": HypergraphData.from_hyperedges("txt", [(0, [1])])},
            }]
        )
        self.assertIsNotNone(model.context_residual_projector)
        assert model.context_residual_projector is not None
        self.assertTrue(torch.equal(
            model.context_residual_projector.weight,
            torch.zeros_like(model.context_residual_projector.weight),
        ))

        output = model(**batch)
        output.loss.backward()
        self.assertIsNotNone(model.context_residual_projector.weight.grad)

    def test_soft_prompt_injection_supports_batches(self) -> None:
        processor = build_processor()
        model = build_model(processor)
        soft_prompt_id = processor.get_token_id_map()["soft_prompt_token_id"]
        input_ids = torch.full((2, 2), soft_prompt_id, dtype=torch.long)
        inputs_embeds = torch.zeros(2, 2, 16)

        injected = model._inject_special_embeddings(input_ids, inputs_embeds)
        expected = model.soft_prompt_embeddings.weight.unsqueeze(0).expand(2, -1, -1)

        self.assertTrue(torch.equal(injected, expected))
        injected.sum().backward()
        self.assertIsNotNone(model.soft_prompt_embeddings.weight.grad)

    def test_semantic_view_initializes_its_feature_table(self) -> None:
        processor = build_processor()
        backbone_config = GPT2Config(
            vocab_size=len(processor.tokenizer),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=128,
            bos_token_id=None,
            eos_token_id=1,
            pad_token_id=1,
        )
        graph_config = HoCRSHypergraphConfig(
            input_dim=8, hidden_dim=8, output_dim=8, num_layers=1
        )
        config = HoCRSConfig(
            backbone_config=backbone_config,
            views=["txt"],
            txt_hypergraph_config=graph_config,
            num_items=3,
            item_dim=8,
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=2,
            freeze_backbone=True,
            train_special_tokens=True,
            **processor.get_token_id_map(),
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
        features = torch.randn(3, 8)
        model.initialize_feature_tables({"txt": features})
        self.assertTrue(torch.equal(model.txt_feature_table, features))

    def test_no_hypergraph_forward_and_backward(self) -> None:
        processor = build_processor()
        backbone_config = GPT2Config(
            vocab_size=len(processor.tokenizer),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=128,
            bos_token_id=None,
            eos_token_id=1,
            pad_token_id=1,
        )
        config = HoCRSConfig(
            backbone_config=backbone_config,
            views=[],
            num_items=3,
            item_dim=8,
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=2,
            freeze_backbone=True,
            train_special_tokens=True,
            **processor.get_token_id_map(),
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
        model.initialize_feature_tables({})
        graphless_batch = HoCRSDataCollator(processor)(
            [
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {},
                }
            ]
        )

        output = model(**graphless_batch)
        self.assertEqual(model.config.views, ())
        self.assertEqual(output.rec_scores.shape, (1, 3))
        self.assertEqual(model.recommendation_head.item_table.weight.shape, (3, 8))
        output.loss.backward()
        self.assertIsNotNone(model.recommendation_head.user_projector[0].weight.grad)
        no_graph_output = model(
            input_ids=graphless_batch["input_ids"],
            attention_mask=graphless_batch["attention_mask"],
            labels=graphless_batch["labels"],
            rec_labels=graphless_batch["rec_labels"],
            hypergraphs=None,
        )
        self.assertEqual(no_graph_output.rec_scores.shape, (1, 3))

    def test_semantic_hybrid_uses_content_for_unseen_and_gates_id_residual(self) -> None:
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
            input_dim=8, hidden_dim=8, output_dim=8, num_layers=1
        )
        config = HoCRSConfig(
            backbone_config=backbone_config,
            views=["txt", "img"],
            txt_hypergraph_config=graph_config,
            img_hypergraph_config=graph_config,
            num_items=3,
            item_dim=4,
            item_table_mode="semantic_hybrid",
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=0,
            **processor.get_token_id_map(),
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
        tables = {
            "txt": torch.randn(3, 8),
            "img": torch.randn(3, 8),
        }
        model.initialize_feature_tables(
            tables,
            item_frequencies=torch.tensor([0.0, 1.0, 9.0]),
        )
        head = model.recommendation_head
        before = head.item_embeddings(tables).detach().clone()
        with torch.no_grad():
            head.item_residual.weight.fill_(10.0)
        after = head.item_embeddings(tables)

        self.assertEqual(float(head.item_frequency_gate[0]), 0.0)
        self.assertEqual(float(head.item_frequency_gate[2]), 1.0)
        torch.testing.assert_close(after[0], before[0])
        self.assertFalse(torch.allclose(after[1], before[1]))

        loss = -after[0, 0] - after[1, 0]
        loss.backward()
        self.assertTrue(torch.equal(
            head.item_residual.weight.grad[0],
            torch.zeros_like(head.item_residual.weight.grad[0]),
        ))
        self.assertGreater(float(head.item_residual.weight.grad[1].abs().sum()), 0.0)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = HoCRSModel.from_pretrained(directory)
        torch.testing.assert_close(
            restored.recommendation_head.item_frequency_gate,
            head.item_frequency_gate,
        )
        torch.testing.assert_close(
            restored.recommendation_head.item_embeddings(
                {
                    view: getattr(restored, f"{view}_feature_table")
                    for view in restored.config.views
                }
            ),
            head.item_embeddings(tables),
        )

    def test_collator_keeps_graph_order_and_original_rows_with_text_only_sample(self) -> None:
        processor = build_processor()
        first = HypergraphData.from_hyperedges("txt", [(0, [1])])
        third = HypergraphData.from_hyperedges("txt", [(2, [1])])

        batch = HoCRSDataCollator(processor)(
            [
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {"txt": first},
                },
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {},
                },
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {"txt": third},
                },
            ]
        )

        graph_batch = batch["hypergraphs"]["txt"]
        self.assertEqual(graph_batch["node_ids"].tolist(), [0, 1, 2, 1])
        self.assertEqual(graph_batch["node_positions"][:, 0].tolist(), [0, 0, 2, 2])
        self.assertEqual(graph_batch["hyperedge_positions"][:, 0].tolist(), [0, 2])

    def test_all_graphless_batch_with_moe_has_finite_diagnostics(self) -> None:
        processor = build_processor()
        backbone_config = GPT2Config(
            vocab_size=len(processor.tokenizer),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=128,
            bos_token_id=None,
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
            item_dim=8,
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=2,
            use_moe=True,
            moe_num_experts=4,
            freeze_backbone=True,
            train_special_tokens=True,
            **processor.get_token_id_map(),
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
        model.initialize_feature_tables({"txt": torch.randn(3, 8)})
        batch = HoCRSDataCollator(processor)(
            [
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {},
                },
                {
                    "context": "hello",
                    "target_item_id": 2,
                    "response": "reply",
                    "hypergraphs": {},
                },
            ]
        )

        output = model(**batch)

        self.assertEqual(output.moe_usage.shape, (4,))
        self.assertEqual(output.moe_view_usage.shape, (1, 4))
        self.assertTrue(torch.isfinite(output.moe_usage).all())
        self.assertTrue(torch.isfinite(output.moe_view_usage).all())
        self.assertTrue(torch.isfinite(output.moe_router_entropy))
        output.loss.backward()

    def test_direct_projection_skips_hypergraph_encoder(self) -> None:
        processor = build_processor()
        backbone_config = GPT2Config(
            vocab_size=len(processor.tokenizer),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=128,
            bos_token_id=None,
            eos_token_id=1,
            pad_token_id=1,
        )
        graph_config = HoCRSHypergraphConfig(
            input_dim=8, hidden_dim=8, output_dim=4, num_layers=3
        )
        config = HoCRSConfig(
            backbone_config=backbone_config,
            views=["txt"],
            txt_hypergraph_config=graph_config,
            num_items=3,
            item_dim=8,
            use_hypergraph_encoder=False,
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=2,
            freeze_backbone=True,
            train_special_tokens=True,
            **processor.get_token_id_map(),
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
        model.initialize_feature_tables({"txt": torch.randn(3, 8)})
        graph = HypergraphData.from_hyperedges("txt", [(0, [1])])
        batch = HoCRSDataCollator(processor)(
            [
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {"txt": graph},
                }
            ]
        )

        output = model(**batch)

        self.assertFalse(model.config.use_hypergraph_encoder)
        self.assertEqual(len(model.hypergraph_encoders), 0)
        self.assertEqual(
            model.hypergraph_projectors["txt"].node_projector.in_features,
            graph_config.input_dim,
        )
        output.loss.backward()
        self.assertIsNotNone(
            model.hypergraph_projectors["txt"].node_projector.weight.grad
        )

    def test_left_truncation_preserves_graphs_and_response_labels(self) -> None:
        processor = build_processor()
        graph = HypergraphData.from_hyperedges("txt", [(0, [1])])
        batch = HoCRSDataCollator(processor, max_length=50)(
            [
                {
                    "context": " ".join(["hello"] * 100),
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {"txt": graph},
                }
            ]
        )

        self.assertEqual(batch["input_ids"].shape[1], 50)
        self.assertGreater(int((batch["labels"] != -100).sum()), 0)
        self.assertEqual(batch["hypergraphs"]["txt"]["node_positions"].shape[0], 2)

    def test_long_response_is_truncated_after_complete_graph_prompt(self) -> None:
        processor = build_processor()
        graph = HypergraphData.from_hyperedges("txt", [(0, [1])])
        batch = HoCRSDataCollator(processor, max_length=50)(
            [
                {
                    "context": " ".join(["hello"] * 100),
                    "target_item_id": 1,
                    "response": " ".join(["reply"] * 100),
                    "hypergraphs": {"txt": graph},
                }
            ]
        )

        self.assertEqual(batch["input_ids"].shape[1], 50)
        self.assertGreater(int((batch["labels"] != -100).sum()), 0)
        self.assertEqual(batch["hypergraphs"]["txt"]["node_positions"].shape[0], 2)

    def test_collated_forward_backward_and_standard_reload(self) -> None:
        processor = build_processor()
        graph = HypergraphData.from_hyperedges("txt", [(0, [1])])
        batch = HoCRSDataCollator(processor)(
            [
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {"txt": graph},
                }
            ]
        )
        model = build_model(processor)

        output = model(**batch)
        self.assertEqual(output.rec_scores.shape, (1, 3))
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        projector_grad = model.hypergraph_projectors["txt"].node_projector.weight.grad
        self.assertIsNotNone(projector_grad)
        self.assertGreater(float(projector_grad.abs().sum()), 0.0)

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
            reloaded_processor = HoCRSProcessor.from_pretrained(directory)
            reloaded_model = HoCRSModel.from_pretrained(directory)
            auto_model = AutoModelForCausalLM.from_pretrained(directory)

        self.assertEqual(reloaded_processor.num_soft_prompt_tokens, 2)
        self.assertIsInstance(auto_model, HoCRSModel)
        self.assertTrue(
            torch.equal(model.txt_feature_table, reloaded_model.txt_feature_table)
        )
        self.assertTrue(
            torch.equal(
                model.recommendation_head.item_table.weight,
                reloaded_model.recommendation_head.item_table.weight,
            )
        )
        self.assertNotEqual(
            reloaded_model.txt_feature_table.data_ptr(),
            reloaded_model.recommendation_head.item_table.weight.data_ptr(),
        )

    def test_wrapping_does_not_reinitialize_a_pretrained_backbone(self) -> None:
        processor = build_processor()
        backbone_config = GPT2Config(
            vocab_size=len(processor.tokenizer),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=128,
            bos_token_id=None,
            eos_token_id=1,
            pad_token_id=1,
        )
        backbone = GPT2LMHeadModel(backbone_config)
        with torch.no_grad():
            backbone.transformer.wte.weight.fill_(0.125)
        graph_config = HoCRSHypergraphConfig(
            input_dim=8, hidden_dim=8, output_dim=8, num_layers=1
        )
        config = HoCRSConfig(
            backbone_config=backbone_config,
            views=["txt"],
            txt_hypergraph_config=graph_config,
            num_items=3,
            item_dim=8,
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=0,
            train_special_tokens=False,
        )

        model = HoCRSModel(config, backbone)
        self.assertAlmostEqual(
            float(model.backbone.transformer.wte.weight.mean()), 0.125, places=6
        )


if __name__ == "__main__":
    unittest.main()
