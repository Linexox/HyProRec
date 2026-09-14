import unittest

import torch
from hyprorec.scripts.prepare_content_table import build_content_table


class FeatureTableBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = {
            modality: torch.arange(12, dtype=torch.float32).reshape(3, 4) + index
            for index, modality in enumerate(("txt", "img", "ado", "vdo"))
        }

    # START: Verify normalized mean fusion in the shared aligned space.
    def test_content_table_averages_enabled_aligned_modalities(self) -> None:
        full = build_content_table(self.tables, ["txt", "img", "ado", "vdo"])
        partial = build_content_table(self.tables, ["txt", "img"])

        self.assertEqual(full.shape, (3, 4))
        self.assertEqual(partial.shape, (3, 4))
        self.assertTrue(torch.allclose(full.norm(dim=-1), torch.ones(3)))
        self.assertTrue(torch.allclose(partial.norm(dim=-1), torch.ones(3)))

    def test_content_table_respects_modality_mask(self) -> None:
        mask = {
            modality: torch.tensor([True, modality == "txt", True])
            for modality in self.tables
        }

        content = build_content_table(self.tables, ["txt", "img"], mask)
        expected = torch.nn.functional.normalize(self.tables["txt"][1:2], dim=-1)

        self.assertTrue(torch.allclose(content[1:2], expected))

    # START: A strict single-modality ablation must not impute another modality.
    def test_content_table_keeps_missing_items_zero(self) -> None:
        mask = {"img": torch.tensor([False, True, True])}

        content = build_content_table({"img": self.tables["img"]}, ["img"], mask)

        self.assertTrue(torch.equal(content[0], torch.zeros(4)))
        self.assertAlmostEqual(float(content[1].norm()), 1.0, places=6)

    # END: A strict single-modality ablation must not impute another modality.

    def test_content_table_rejects_unaligned_widths(self) -> None:
        tables = {"txt": torch.ones(3, 2), "img": torch.ones(3, 3)}

        with self.assertRaisesRegex(ValueError, "same width"):
            build_content_table(tables, ["txt", "img"])

    # END: Verify normalized mean fusion in the shared aligned space.


if __name__ == "__main__":
    unittest.main()
