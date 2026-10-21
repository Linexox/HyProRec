"""Subset experiments remove omitted modalities from recommendation as well."""
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import GPT2Config

from hyprorec.arguments import DataArguments, ModelArguments
from hyprorec.configuration_hocrs import HoCRSConfig
from hyprorec.model_factory import load_modality_tables
from hyprorec.modeling_hocrs import HoCRSRecommendationModel


class ActiveItemViewsTest(unittest.TestCase):
    views = ("txt", "img", "vdo")

    def test_active_loader_does_not_require_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            embeddings = Path(directory) / "embeddings"
            embeddings.mkdir()
            for view in self.views:
                torch.save(torch.randn(3, 4), embeddings / f"{view}_embeddings.pt")
            data = DataArguments(dataset_path=directory, views=list(self.views))
            tables = load_modality_tables(data, ModelArguments(item_table_view="active"))
        self.assertEqual(tuple(tables), self.views)

    def test_active_checkpoint_restores_three_branches_and_logits(self):
        dimensions = {"txt": 4, "img": 4, "ado": 4, "vdo": 4}
        config = HoCRSConfig(
            backbone_config=GPT2Config(n_embd=16, n_layer=1, n_head=2, vocab_size=32),
            views=self.views,
            item_table_view="active",
            item_feature_dims=dimensions,
            num_items=3,
            recommendation_hidden_dim=8,
            recommendation_attention_dim=4,
            recommendation_router_hidden_dim=8,
            **{
                f"{view}_hypergraph_config": {
                    "input_dim": 4, "hidden_dim": 8, "output_dim": 4, "num_layers": 1,
                }
                for view in self.views
            },
        )
        model = HoCRSRecommendationModel(config).eval()
        model.initialize_feature_tables({v: torch.randn(3, 4) for v in dimensions})
        head = model.recommendation_head
        self.assertEqual(head.names, self.views)
        self.assertEqual(tuple(model._feature_tables()), self.views)
        self.assertEqual(head.router[-1].out_features, 3)
        hidden = torch.randn(2, 5, 16)
        mask = torch.ones(2, 5, dtype=torch.bool)
        labels = torch.tensor([0, 2])
        result = head(hidden, mask, model._feature_tables(), labels)
        result.loss.backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = HoCRSRecommendationModel.from_pretrained(directory)
        self.assertEqual(restored.config.item_views, self.views)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        actual = restored.recommendation_head(hidden, mask, restored._feature_tables(), labels)
        torch.testing.assert_close(actual.logits, result.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual.loss, result.loss, rtol=1e-5, atol=1e-6)

    def test_full_still_keeps_all_branches_with_subset_graphs(self):
        config = HoCRSConfig(
            views=self.views,
            item_table_view="full",
            item_feature_dims={"ado": 4, "vdo": 4, "txt": 4, "img": 4},
        )
        self.assertEqual(config.item_views, ("txt", "img", "ado", "vdo"))


if __name__ == "__main__":
    unittest.main()
