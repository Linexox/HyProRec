import unittest

import torch
import torch.nn.functional as F

from hyprorec.scripts.train import _build_feature_tables


class FeatureTableBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = {
            modality: torch.arange(12, dtype=torch.float32).reshape(3, 4) + index
            for index, modality in enumerate(("txt", "img", "ado", "vdo"))
        }

    def test_item_content_base_uses_enabled_semantic_modalities(self) -> None:
        feature_tables, item_init = _build_feature_tables(
            self.tables, ["co", "txt", "img"]
        )
        expected = F.normalize(
            torch.stack(
                [
                    F.normalize(self.tables[modality], dim=-1)
                    for modality in ("txt", "img")
                ]
            ).mean(dim=0),
            dim=-1,
        )

        self.assertEqual(set(feature_tables), {"txt", "img"})
        self.assertTrue(torch.equal(item_init, expected))

    def test_no_graph_content_base_falls_back_to_all_modalities(self) -> None:
        feature_tables, item_init = _build_feature_tables(self.tables, [])
        expected = F.normalize(
            torch.stack(
                [F.normalize(self.tables[modality], dim=-1) for modality in self.tables]
            ).mean(dim=0),
            dim=-1,
        )

        self.assertEqual(feature_tables, {})
        self.assertTrue(torch.equal(item_init, expected))


if __name__ == "__main__":
    unittest.main()
