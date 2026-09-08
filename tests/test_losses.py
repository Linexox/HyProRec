import unittest

import torch

from hyprorec.losses import multi_positive_contrastive_loss


class ContrastiveLossTest(unittest.TestCase):
    def test_repeated_item_ids_are_multi_positive(self) -> None:
        features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        item_ids = torch.tensor([4, 4, 8])

        loss = multi_positive_contrastive_loss(features, features, item_ids, 0.1)

        self.assertIsNotNone(loss)
        self.assertLess(float(loss), 0.01)

    def test_batch_without_negatives_is_skipped(self) -> None:
        features = torch.ones(2, 3)

        loss = multi_positive_contrastive_loss(
            features, features, torch.tensor([1, 1]), 0.1
        )

        self.assertIsNone(loss)


if __name__ == "__main__":
    unittest.main()
