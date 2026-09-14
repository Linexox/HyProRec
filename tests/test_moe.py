import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from hyprorec.configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from hyprorec.modeling_hocrs import GraphTokenMoE, HoCRSModel, UserProjectionMoE


class GraphTokenMoETest(unittest.TestCase):
    def test_zero_initialized_experts_preserve_projected_features(self) -> None:
        moe = GraphTokenMoE(
            hidden_size=8,
            views=("txt", "img"),
            num_experts=2,
            expert_hidden_size=4,
            router_temperature=1.0,
            residual_scale_init=1.0,
        )
        moe.reset_output_layers()
        features = torch.randn(5, 8)
        output, weights, delta = moe(features, "txt")

        self.assertTrue(torch.allclose(output, features))
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(5)))
        self.assertFalse(torch.isnan(weights).any())

    def test_user_moe_preserves_base_projection_at_initialization(self) -> None:
        moe = UserProjectionMoE(
            input_dim=8,
            output_dim=4,
            num_experts=2,
            expert_hidden_dim=4,
            router_temperature=1.0,
            residual_scale_init=1.0,
        )
        hidden_states = torch.randn(3, 8)
        base_users = torch.randn(3, 4)
        output, weights, delta = moe(hidden_states, base_users)

        self.assertTrue(torch.allclose(output, base_users))
        self.assertTrue(torch.allclose(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3)))

    def test_moe_mode_creates_router_and_experts(self) -> None:
        backbone_config = GPT2Config(
            vocab_size=32,
            n_embd=8,
            n_layer=1,
            n_head=2,
            n_positions=64,
        )
        graph_config = HoCRSHypergraphConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=4,
            num_layers=1,
        )
        config = HoCRSConfig(
            backbone_config=backbone_config,
            views=["txt"],
            txt_hypergraph_config=graph_config,
            num_items=3,
            item_dim=4,
            recommendation_hidden_dim=4,
            use_moe=True,
            moe_num_experts=2,
            moe_hidden_dim=4,
        )
        model = HoCRSModel(config, GPT2LMHeadModel(backbone_config))

        self.assertIsNotNone(model.moe)
        self.assertEqual(model.moe.view_embeddings.shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
