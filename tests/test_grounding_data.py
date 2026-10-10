import tempfile
import unittest
from pathlib import Path

import torch

from hyprorec.data.grounding import GroundingSourceDataset, HoCRSGroundingCollator
from hyprorec.data.hypergraph import HypergraphData


class GroundingDataTest(unittest.TestCase):
    def test_missing_raw_modality_uses_offline_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "movies_info.csv").write_text(
                "movieName,description\nA,first\nB,second\nC,third\n",
                encoding="utf-8",
            )
            source = GroundingSourceDataset(path, ("img",))
            graph = HypergraphData.from_hyperedges("img", [(0, [1])])
            collator = HoCRSGroundingCollator(
                {"img": torch.randn(3, 4)}, ("img",), source
            )

            batch = collator([{"hypergraphs": {"img": graph}}])

        self.assertIn("feature_values", batch["source_data"]["img"])
        self.assertEqual(batch["source_data"]["img"]["feature_values"].shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
