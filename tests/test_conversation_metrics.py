import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.trainer_utils import EvalPrediction

from hyprorec.arguments import DataArguments
from hyprorec.evaluation import build_compute_metrics
from hyprorec.metrics import (
    conversation_teacher_forcing_metrics,
    conversation_teacher_forcing_metrics_from_texts,
)


class ConversationMetricsTest(unittest.TestCase):
    def setUp(self):
        vocabulary = {token: i for i, token in enumerate(
            ["[UNK]", "[EOS]", "我", "爱", "看", "电", "影", "视", "也", "A", "I"]
        )}
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
        backend.decoder = decoders.Fuse()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
        )

    def metrics(self, references, hypotheses, mode="jieba"):
        return conversation_teacher_forcing_metrics_from_texts(
            references, hypotheses, metric_tokenization=mode
        )

    def test_english_restores_whitespace_ngram_counts(self):
        metrics = self.metrics(
            ["one two three four five"], ["one two three four six"], "whitespace"
        )
        self.assertAlmostEqual(metrics["bleu@1"], 4 / 5)
        self.assertAlmostEqual(metrics["bleu@4"], (1 / 5) ** 0.25)
        self.assertEqual(metrics["dist@4"], 1.0)

    def test_word_segmentation_keeps_punctuation_and_filters_whitespace(self):
        from hyprorec.metrics import _tokens

        self.assertEqual(_tokens("电影，电视。", "jieba"), ["电影", "，", "电视", "。"])
        self.assertEqual(_tokens(" 电影  电视\n", "jieba"), ["电影", "电视"])
        self.assertEqual(_tokens("Movie, IS good.", "whitespace"), ["Movie,", "IS", "good."])

    def test_metric_mode_is_validated_in_data_arguments(self):
        self.assertEqual(DataArguments().metric_tokenization, "whitespace")
        self.assertEqual(DataArguments(metric_tokenization="jieba").metric_tokenization, "jieba")
        with self.assertRaises(ValueError):
            DataArguments(metric_tokenization="backbone")

    def test_unspaced_chinese_exact_match_has_all_bleu_orders(self):
        metrics = self.metrics(["我爱看电影也爱看电视"], ["我爱看电影也爱看电视"])
        for n in range(1, 5):
            self.assertAlmostEqual(metrics[f"bleu@{n}"], 1.0)

    def test_short_chinese_sentence_keeps_unsmoothed_bleu_rule(self):
        metrics = self.metrics(["我爱看电影"], ["我爱看电影"])
        self.assertEqual(metrics["bleu@1"], 1.0)
        self.assertEqual(metrics["bleu@4"], 0.0)

    def test_partial_chinese_match_has_nonzero_high_order_bleu(self):
        metrics = self.metrics(
            ["我爱看电影也爱看电视"], ["我爱看电影也爱看电影"]
        )
        self.assertGreater(metrics["bleu@4"], 0.0)
        self.assertLess(metrics["bleu@4"], 1.0)

    def test_distinct_counts_words_and_keeps_sentence_boundaries(self):
        metrics = self.metrics(["电影电影"] * 2, ["电影电影"] * 2)
        self.assertAlmostEqual(metrics["dist@1"], 1 / 4)
        self.assertAlmostEqual(metrics["dist@2"], 1 / 2)
        self.assertEqual(metrics["dist@3"], 0.0)

    def test_mixed_chinese_and_english_exact_match(self):
        metrics = self.metrics(["我爱看AI电影"], ["我爱看AI电影"])
        self.assertEqual(metrics["bleu@4"], 1.0)

    def test_empty_hypothesis_and_empty_corpus(self):
        for references, hypotheses in [(["我爱看电影"], [""]), ([], [])]:
            with self.subTest(references=references):
                self.assertTrue(all(v == 0.0 for v in self.metrics(references, hypotheses).values()))

    def test_eval_filters_special_tokens_and_preserves_response_count(self):
        response = self.tokenizer("我爱看电影也爱看电视", add_special_tokens=False)["input_ids"]
        response.append(self.tokenizer.eos_token_id)
        labels = np.array([[-100, *response, -100]])
        predictions = np.array([[*response, -100]])
        result = build_compute_metrics(self.tokenizer, metric_tokenization="jieba")(EvalPrediction(
            predictions=(np.array([[1, 0]]), predictions),
            label_ids=(labels, np.array([1])),
        ))
        self.assertEqual(result["bleu@4"], 1.0)
        self.assertEqual(result["response_tokens"], len(response))
        self.assertEqual(result["recall@1"], 1.0)
        self.assertAlmostEqual(result["dist@1"], 5 / 6)

        logits = torch.zeros(1, labels.shape[1], len(self.tokenizer))
        for position, token in enumerate(response):
            logits[0, position, token] = 10.0
        direct = conversation_teacher_forcing_metrics(
            logits, torch.tensor(labels), self.tokenizer, metric_tokenization="jieba"
        )
        for name, value in direct.items():
            self.assertEqual(result[name], value)

    def test_eval_word_metrics_do_not_depend_on_backbone_token_boundaries(self):
        vocabulary = {
            token: i for i, token in enumerate(
                ["[UNK]", "[EOS]", "我", "爱看", "电影", "也", "电视"]
            )
        }
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.decoder = decoders.Fuse()
        word_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]"
        )
        text = "我爱看电影也爱看电视"
        character_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        word_ids = [vocabulary[word] for word in ["我", "爱看", "电影", "也", "爱看", "电视"]]
        results = []
        counts = []
        for tokenizer, response in [(self.tokenizer, character_ids), (word_tokenizer, word_ids)]:
            response = [*response, tokenizer.eos_token_id]
            result = build_compute_metrics(tokenizer, "jieba")(EvalPrediction(
                predictions=(np.array([[1, 0]]), np.array([response])),
                label_ids=(np.array([[-100, *response]]), np.array([1])),
            ))
            counts.append(result.pop("response_tokens"))
            results.append(result)
        self.assertEqual(results[0], results[1])
        self.assertNotEqual(counts[0], counts[1])


if __name__ == "__main__":
    unittest.main()
