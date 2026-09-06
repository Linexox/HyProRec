import json
import tempfile
import unittest
from pathlib import Path

import torch

from hyprorec.data.batch import BatchData, batch_hypergraphs
from hyprorec.data.hypergraph import HypergraphData, HypergraphTable
from hyprorec.scripts.prepare_hyperedge_table import compute_cooccurrence_neighbors


class HypergraphTest(unittest.TestCase):
    def test_cooccurrence_uses_unique_items_per_training_dialogue(self) -> None:
        conversations = [
            {
                "dialog": [
                    {"items": [0, 1, 1]},
                    {"items": [1, 2]},
                ]
            },
            {"dialog": [{"items": [0, 1]}]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "train_data.json").write_text(
                json.dumps(conversations), encoding="utf-8"
            )
            neighbors = compute_cooccurrence_neighbors(path, num_items=4)

        self.assertEqual(neighbors[0], [1, 2])
        self.assertEqual(neighbors[1], [0, 2])
        self.assertEqual(neighbors[2], [0, 1])
        self.assertEqual(neighbors[3], [])

    def test_local_retrieval_and_disjoint_batch(self) -> None:
        table = HypergraphTable(
            {
                "co": [
                    [0, 1, 2],
                    [1, 0, 2],
                    [2, 0, 1],
                ]
            }
        )
        first = table.build_local([0], view="co", topk=1, khop=2)
        second = table.build_local([2], view="co", topk=1, khop=1)
        batch = batch_hypergraphs([first, second])

        self.assertEqual(first.num_hyperedges, 2)
        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch["node_ptr"].tolist(), [0, 2, 4])
        self.assertEqual(batch["edge_ptr"].tolist(), [0, 2, 3])
        self.assertGreaterEqual(int(batch["hyperedge_index"][1, -1]), 2)

    def test_batch_device_transfer_is_recursive_and_non_mutating(self) -> None:
        original = BatchData(
            {"input_ids": torch.tensor([1]), "nested": {"value": torch.tensor([2])}}
        )
        moved = original.to("cpu")

        self.assertIsNot(original, moved)
        self.assertIsNot(original["nested"], moved["nested"])
        self.assertEqual(moved["nested"]["value"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
