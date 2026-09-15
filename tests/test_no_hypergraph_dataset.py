import json
import tempfile
import unittest
from pathlib import Path

from hyprorec.data.redial import HoCRSDataset, HoCRSDatasetConfig


class NoHypergraphDatasetTest(unittest.TestCase):
    def test_dataset_keeps_recommendations_without_context_items(self) -> None:
        conversations = [
            {
                "dialog": [
                    {"role": "Recommender", "text": "cold reply", "items": [1]},
                    {"role": "Seeker", "text": "I like it", "items": [1]},
                    {"role": "Recommender", "text": "warm reply", "items": [2]},
                ]
            }
        ]

        samples = HoCRSDataset._build_samples(conversations)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["context_item_ids"], [])
        self.assertEqual(samples[1]["context_item_ids"], [1])

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
