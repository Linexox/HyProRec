import unittest

import torch
from hyprorec.scripts.prepare_content_table import build_content_table


class FeatureTableBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tables = {
            modality: torch.arange(12, dtype=torch.float32).reshape(3, 4) + index
            for index, modality in enumerate(("txt", "img", "ado", "vdo"))
        }

    # START: Verify fixed-width slot concatenation for modality ablations.
    def test_content_table_uses_enabled_slots_without_changing_width(self) -> None:
        full = build_content_table(self.tables, ["txt", "img", "ado", "vdo"])
        partial = build_content_table(self.tables, ["txt", "img"])

        self.assertEqual(full.shape, (3, 16))
        self.assertEqual(partial.shape, full.shape)
        self.assertGreater(torch.count_nonzero(partial[:, :8]).item(), 0)
        self.assertEqual(torch.count_nonzero(partial[:, 8:]).item(), 0)

    def test_content_table_supports_different_encoder_widths(self) -> None:
        tables = {
            modality: torch.ones(3, width)
            for modality, width in zip(("txt", "img", "ado", "vdo"), (2, 3, 4, 5))
        }

        content = build_content_table(tables, ["ado"])

        self.assertEqual(content.shape, (3, 14))
        self.assertEqual(torch.count_nonzero(content[:, :5]).item(), 0)
        self.assertGreater(torch.count_nonzero(content[:, 5:9]).item(), 0)
        self.assertEqual(torch.count_nonzero(content[:, 9:]).item(), 0)

    # END: Verify fixed-width slot concatenation for modality ablations.


if __name__ == "__main__":
    unittest.main()
