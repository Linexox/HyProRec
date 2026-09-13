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
from hyprorec.modeling_hocrs import HoCRSModel, HypergraphEncoderOutput
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
        views=["co"],
        co_hypergraph_config=graph_config,
        num_items=3,
        item_dim=8,
        recommendation_hidden_dim=8,
        num_soft_prompt_tokens=2,
        freeze_backbone=True,
        train_special_tokens=True,
        **processor.get_token_id_map(),
    )
    model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
    model.initialize_feature_tables({"co": torch.randn(3, 8)})
    return model


class HoCRSModelTest(unittest.TestCase):
    def build_gate_model(
        self,
        processor: HoCRSProcessor,
        *,
        use_sample_conditional_gate: bool = True,
        use_residual_injection: bool = False,
    ) -> HoCRSModel:
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
            item_dim=8,
            use_hypergraph_encoder=False,
            use_sample_conditional_gate=use_sample_conditional_gate,
            use_residual_injection=use_residual_injection,
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=2,
            freeze_backbone=True,
            train_special_tokens=True,
            **processor.get_token_id_map(),
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
        model.initialize_feature_tables(
            {"txt": torch.randn(3, 8), "img": torch.randn(3, 8)},
            item_table_init=torch.randn(3, 8),
        )
        return model

    def test_sample_gate_is_per_sample_and_per_view_and_starts_at_one(self) -> None:
        processor = build_processor()
        model = self.build_gate_model(processor)
        self.assertEqual(set(model.sample_gate_networks), {"txt", "img"})
        self.assertEqual(model._compute_sample_gate(
            torch.randn(2, 16), torch.randn(2, 16), "txt"
        ).shape, (2, 1))
        self.assertTrue(torch.equal(
            model._compute_sample_gate(
                torch.randn(2, 16), torch.randn(2, 16), "txt"
            ),
            torch.ones(2, 1),
        ))
        with torch.no_grad():
            model.sample_gate_networks["txt"][-1].weight.fill_(0.25)
            model.sample_gate_networks["txt"][-1].bias.zero_()
            model.sample_gate_networks["img"][-1].weight.fill_(-0.25)
            model.sample_gate_networks["img"][-1].bias.zero_()
        context = torch.ones(2, 16)
        graph = torch.stack((torch.ones(16), torch.full((16,), 2.0)))
        txt = model._compute_sample_gate(context, graph, "txt")
        img = model._compute_sample_gate(context, graph, "img")
        self.assertFalse(torch.allclose(txt, img))
        self.assertFalse(torch.allclose(txt[0], txt[1]))

    def test_gate_context_excludes_response_and_graph_positions(self) -> None:
        processor = build_processor()
        model = self.build_gate_model(processor)
        embeddings = torch.arange(1, 1 + 5 * 2, dtype=torch.float32).view(1, 5, 2)
        graph = {
            "node_positions": torch.tensor([[0, 1]]),
            "hyperedge_positions": torch.tensor([[0, 3]]),
        }
        labels = torch.tensor([[-100, -100, -100, 7, 8]])
        summary = model._pool_gate_context(
            embeddings, {"txt": graph}, torch.ones(1, 5), labels
        )
        expected = embeddings[:, [0, 2], :].mean(dim=1)
        self.assertTrue(torch.equal(summary, expected))

    def test_residual_injection_preserves_placeholder_embedding(self) -> None:
        processor = build_processor()
        model = self.build_gate_model(
            processor,
            use_sample_conditional_gate=False,
            use_residual_injection=True,
        )
        batch = HoCRSDataCollator(processor)(
            [{
                "context": "hello",
                "target_item_id": 1,
                "response": "reply",
                "hypergraphs": {
                    "txt": HypergraphData.from_hyperedges("txt", [(0, [1])]),
                    "img": HypergraphData.from_hyperedges("img", [(0, [1])]),
                },
            }]
        )
        base = torch.zeros(1, batch["input_ids"].size(1), 16)
        projected = HypergraphEncoderOutput(
            torch.ones(2, 16), torch.full((1, 16), 2.0)
        )
        with mock.patch.object(
            model.hypergraph_projectors["txt"], "forward", return_value=projected
        ), mock.patch.object(
            model.hypergraph_projectors["img"], "forward", return_value=projected
        ):
            injected, _ = model._inject_hypergraphs(
                base, batch["hypergraphs"], batch["attention_mask"], batch["labels"]
            )
        txt_node = batch["hypergraphs"]["txt"]["node_positions"][0]
        txt_edge = batch["hypergraphs"]["txt"]["hyperedge_positions"][0]
        self.assertTrue(torch.allclose(injected[tuple(txt_node)], torch.full((16,), 0.1)))
        self.assertTrue(torch.allclose(injected[tuple(txt_edge)], torch.full((16,), 0.2)))

    def test_new_configuration_fields_round_trip(self) -> None:
        processor = build_processor()
        model = self.build_gate_model(processor, use_residual_injection=True)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = HoCRSModel.from_pretrained(directory)
        self.assertTrue(reloaded.config.use_sample_conditional_gate)
        self.assertTrue(reloaded.config.use_residual_injection)
        self.assertEqual(reloaded.config.gate_hidden_dim, 128)
        self.assertEqual(reloaded.config.residual_scale_init, 0.1)

    # START: Verify shared content initialization does not tie trainable tables.
    def test_co_and_item_tables_copy_the_same_content_independently(self) -> None:
        processor = build_processor()
        model = build_model(processor)
        content = torch.randn(3, 8)

        model.initialize_feature_tables({"co": content}, item_table_init=content)

        self.assertTrue(torch.equal(model.co_feature_table, content))
        self.assertTrue(
            torch.equal(model.recommendation_head.item_table.weight, content)
        )
        self.assertNotEqual(
            model.co_feature_table.data_ptr(),
            model.recommendation_head.item_table.weight.data_ptr(),
        )

    # END: Verify shared content initialization does not tie trainable tables.

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

    def test_semantic_view_does_not_require_co_view(self) -> None:
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
        model.initialize_feature_tables(
            {"txt": torch.randn(3, 8)}, item_table_init=torch.randn(3, 8)
        )
        self.assertFalse(hasattr(model, "co_feature_table"))

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
        content = torch.randn(3, 8)
        model.initialize_feature_tables({}, item_table_init=content)
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
        self.assertTrue(
            torch.equal(model.recommendation_head.item_table.weight, content)
        )
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
            views=["co"],
            co_hypergraph_config=graph_config,
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
        model.initialize_feature_tables({"co": torch.randn(3, 8)})
        graph = HypergraphData.from_hyperedges("co", [(0, [1])])
        batch = HoCRSDataCollator(processor)(
            [
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {"co": graph},
                }
            ]
        )

        output = model(**batch)

        self.assertFalse(model.config.use_hypergraph_encoder)
        self.assertEqual(len(model.hypergraph_encoders), 0)
        self.assertEqual(
            model.hypergraph_projectors["co"].node_projector.in_features,
            graph_config.input_dim,
        )
        output.loss.backward()
        self.assertIsNotNone(
            model.hypergraph_projectors["co"].node_projector.weight.grad
        )

    # START: Verify the v3 Grounding branch and shared source projector gradient path.
    def test_joint_grounding_updates_graph_and_source_projectors(self) -> None:
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
            views=["txt"],
            txt_hypergraph_config=graph_config,
            num_items=3,
            item_dim=8,
            use_source_projector=True,
            grounding_weight=0.5,
            grounding_ga_sa_weight=1.0,
            grounding_ga_sn_weight=1.0,
            grounding_sa_sn_weight=1.0,
            recommendation_hidden_dim=8,
            num_soft_prompt_tokens=2,
            freeze_backbone=True,
            **processor.get_token_id_map(),
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))
        model.initialize_feature_tables(
            {"txt": torch.randn(3, 8)}, item_table_init=torch.randn(3, 8)
        )
        self.assertTrue(torch.equal(model.source_projector.weight, torch.eye(8)))
        batch = HoCRSDataCollator(processor)(
            [
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {
                        "txt": HypergraphData.from_hyperedges("txt", [(0, [1])])
                    },
                },
                {
                    "context": "hello",
                    "target_item_id": 0,
                    "response": "reply",
                    "hypergraphs": {
                        "txt": HypergraphData.from_hyperedges("txt", [(2, [0])])
                    },
                },
            ]
        )

        output = model(**batch)
        expected = (
            config.beta * output.rec_loss
            + (1.0 - config.beta) * output.conv_loss
            + config.grounding_weight * output.grounding_loss
        )
        self.assertTrue(torch.allclose(output.loss, expected))
        self.assertGreater(float(output.grounding_loss), 0.0)
        output.loss.backward()
        self.assertIsNotNone(model.source_projector.weight.grad)
        self.assertIsNotNone(model.hypergraph_encoders["txt"].layers[0].weight.grad)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = HoCRSModel.from_pretrained(directory)
        self.assertTrue(reloaded.config.use_source_projector)
        self.assertTrue(
            torch.equal(model.source_projector.weight, reloaded.source_projector.weight)
        )

    def test_sa_sn_requires_trainable_source_projector(self) -> None:
        with self.assertRaisesRegex(ValueError, "source projector"):
            HoCRSConfig(
                views=["txt"],
                grounding_weight=1.0,
                grounding_sa_sn_weight=1.0,
                use_source_projector=False,
            )

    # END: Verify the v3 Grounding branch and shared source projector gradient path.

    def test_left_truncation_preserves_graphs_and_response_labels(self) -> None:
        processor = build_processor()
        graph = HypergraphData.from_hyperedges("co", [(0, [1])])
        batch = HoCRSDataCollator(processor, max_length=50)(
            [
                {
                    "context": " ".join(["hello"] * 100),
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {"co": graph},
                }
            ]
        )

        self.assertEqual(batch["input_ids"].shape[1], 50)
        self.assertGreater(int((batch["labels"] != -100).sum()), 0)
        self.assertEqual(batch["hypergraphs"]["co"]["node_positions"].shape[0], 2)

    def test_long_response_is_truncated_after_complete_graph_prompt(self) -> None:
        processor = build_processor()
        graph = HypergraphData.from_hyperedges("co", [(0, [1])])
        batch = HoCRSDataCollator(processor, max_length=50)(
            [
                {
                    "context": " ".join(["hello"] * 100),
                    "target_item_id": 1,
                    "response": " ".join(["reply"] * 100),
                    "hypergraphs": {"co": graph},
                }
            ]
        )

        self.assertEqual(batch["input_ids"].shape[1], 50)
        self.assertGreater(int((batch["labels"] != -100).sum()), 0)
        self.assertEqual(batch["hypergraphs"]["co"]["node_positions"].shape[0], 2)

    def test_collated_forward_backward_and_standard_reload(self) -> None:
        processor = build_processor()
        graph = HypergraphData.from_hyperedges("co", [(0, [1])])
        batch = HoCRSDataCollator(processor)(
            [
                {
                    "context": "hello",
                    "target_item_id": 1,
                    "response": "reply",
                    "hypergraphs": {"co": graph},
                }
            ]
        )
        model = build_model(processor)

        output = model(**batch)
        self.assertEqual(output.rec_scores.shape, (1, 3))
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        projector_grad = model.hypergraph_projectors["co"].node_projector.weight.grad
        self.assertIsNotNone(projector_grad)
        self.assertGreater(float(projector_grad.abs().sum()), 0.0)

        generation_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "hypergraphs": batch["hypergraphs"],
        }
        projector = model.hypergraph_projectors["co"]
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
            torch.equal(model.co_feature_table, reloaded_model.co_feature_table)
        )
        self.assertTrue(
            torch.equal(
                model.recommendation_head.item_table.weight,
                reloaded_model.recommendation_head.item_table.weight,
            )
        )
        self.assertNotEqual(
            reloaded_model.co_feature_table.data_ptr(),
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
            views=["co"],
            co_hypergraph_config=graph_config,
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
