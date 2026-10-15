import csv
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoModel, BertConfig, BertTokenizerFast

from hyprorec.data.grounding import GroundingSourceDataset, HoCRSGroundingCollator
from hyprorec.data.hypergraph import HypergraphData
from hyprorec.modeling_grounding import build_source_config


class GroundingTextTest(unittest.TestCase):
    def test_chinese_source_preserves_bert_vocabulary_and_special_tokens(self):
        pretrained = BertConfig(vocab_size=9, pad_token_id=0)
        config = build_source_config("txt", text_config=pretrained)

        self.assertEqual(config.model_type, "bert")
        self.assertEqual(config.vocab_size, 9)
        self.assertEqual(config.pad_token_id, 0)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.num_hidden_layers, 4)
        self.assertEqual(config.num_attention_heads, 4)
        self.assertEqual(pretrained.hidden_size, 768)
        self.assertEqual(build_source_config("txt").model_type, "mpnet")

    def test_long_chinese_titles_fit_lightweight_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "vocab.txt").write_text(
                "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n神\n笔\n我\n知\n道\n",
                encoding="utf-8",
            )
            tokenizer = BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
            with (path / "movies_info.csv").open("w", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["movieName"])
                writer.writerows([["神笔" * 100], ["我知道"]])
            source = GroundingSourceDataset(path, ("txt",))
            graph = HypergraphData.from_hyperedges("txt", [(0, [1])])
            collator = HoCRSGroundingCollator(
                {"txt": torch.randn(2, 768)}, ("txt",), source, tokenizer,
                text_max_length=64,
            )
            data = collator([{"hypergraphs": {"txt": graph}}])["source_data"]["txt"]

        self.assertEqual(data["input_ids"].size(1), 64)
        self.assertFalse(data["input_ids"].eq(tokenizer.unk_token_id).any())
        config = BertConfig(
            vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=64, pad_token_id=0,
        )
        model = AutoModel.from_config(config).eval()
        with torch.inference_mode():
            hidden = model(**data).last_hidden_state
        self.assertTrue(torch.isfinite(hidden).all())


if __name__ == "__main__":
    unittest.main()
