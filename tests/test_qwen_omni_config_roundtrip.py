"""Standalone Omni thinker configs reload on the LAB Transformers version."""
import json
import unittest

import torch
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig

from hyprorec.backbones.base import ADAPTERS, QwenOmniAdapter
from hyprorec.configuration_hocrs import HoCRSConfig


class QwenOmniConfigRoundtripTest(unittest.TestCase):
    def test_saved_thinker_retains_nested_text_dimensions(self):
        for hidden_size in (2048, 3584):
            with self.subTest(hidden_size=hidden_size):
                thinker = Qwen2_5OmniThinkerConfig(text_config={
                    'hidden_size': hidden_size, 'vocab_size': 151682,
                    'num_hidden_layers': 1, 'intermediate_size': 64,
                    'num_attention_heads': 16, 'num_key_value_heads': 2,
                })
                config = HoCRSConfig(backbone_config=thinker)
                payload = json.loads(config.to_json_string())
                restored = HoCRSConfig.from_dict(payload)
                self.assertIsInstance(restored.backbone_config, Qwen2_5OmniThinkerConfig)
                self.assertEqual(restored.backbone_config.text_config.hidden_size, hidden_size)
                self.assertEqual(restored.backbone_config.text_config.vocab_size, 151682)

    def test_saved_thinker_selects_omni_adapter(self):
        self.assertIs(ADAPTERS.get('qwen2_5_omni_thinker'), QwenOmniAdapter)

    def test_distinct_input_and_output_vocabularies_survive_resize_and_reload(self):
        config = Qwen2_5OmniThinkerConfig(
            text_config={'vocab_size': 32, 'hidden_size': 16, 'intermediate_size': 32,
                         'num_hidden_layers': 1, 'num_attention_heads': 2, 'num_key_value_heads': 2},
            output_vocab_size=40,
        )
        with torch.device('meta'):
            adapter = QwenOmniAdapter.from_config(config)
            adapter.resize_token_embeddings(32)
        self.assertEqual(adapter.get_input_embeddings().num_embeddings, 32)
        self.assertEqual(adapter.get_output_embeddings().out_features, 40)
        saved = HoCRSConfig(backbone_config=adapter.config)
        restored = HoCRSConfig.from_dict(json.loads(saved.to_json_string()))
        self.assertEqual(restored.backbone_config.output_vocab_size, 40)
        with torch.device('meta'):
            loaded = QwenOmniAdapter.from_config(restored.backbone_config)
        self.assertEqual(loaded.get_input_embeddings().num_embeddings, 32)
        self.assertEqual(loaded.get_output_embeddings().out_features, 40)


if __name__ == '__main__':
    unittest.main()
