"""Standalone Omni thinker configs reload on the LAB Transformers version."""
import json
import unittest

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


if __name__ == '__main__':
    unittest.main()
