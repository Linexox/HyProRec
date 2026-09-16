"""Tests for recommendation ranking export metadata."""

from hyprorec.scripts.export_rec_predictions import build_recommendation_metadata


def test_metadata_matches_dataset_samples_with_and_without_history() -> None:
    conversations = [
        {
            "conv_id": "dialog-1",
            "user_id": 7,
            "dialog": [
                {
                    "role": "Recommender",
                    "utt_id": 10,
                    "text": "first",
                    "items": [4],
                },
                {
                    "role": "Seeker",
                    "utt_id": 11,
                    "text": "reply",
                    "items": [4, 4],
                },
                {
                    "role": "Recommender",
                    "utt_id": 12,
                    "text": "second",
                    "items": [8, 9],
                },
            ],
        }
    ]

    rows = build_recommendation_metadata(conversations)

    assert [row["target"] for row in rows] == [4, 8, 9]
    assert rows[0]["history_item_ids"] == []
    assert rows[1]["history_item_ids"] == [4]
    assert rows[2]["history_item_ids"] == [4]
    assert [row["sample_id"] for row in rows] == [
        "dialog-1:10:0",
        "dialog-1:12:0",
        "dialog-1:12:1",
    ]
