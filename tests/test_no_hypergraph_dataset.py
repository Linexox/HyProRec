import json
import tempfile
import unittest
from pathlib import Path

from hyprorec.data.redial import HoCRSDataset, HoCRSDatasetConfig


class NoHypergraphDatasetTest(unittest.TestCase):
    def test_no_hypergraph_dataset_does_not_open_topology_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            conversation = [
                {
                    "dialog": [
                        {"role": "Seeker", "text": "hello", "items": [0]},
                        {"role": "Recommender", "text": "reply", "items": [1]},
                    ]
                }
            ]
            (path / "train_data.json").write_text(
                json.dumps(conversation), encoding="utf-8"
            )
            dataset = HoCRSDataset(
                HoCRSDatasetConfig(dataset_path=str(path), views=()),
                "train",
            )

            self.assertIsNone(dataset.hypergraph_table)
            self.assertEqual(dataset[0]["hypergraphs"], {})


if __name__ == "__main__":
    unittest.main()
