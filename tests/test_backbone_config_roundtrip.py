"""Composite checkpoint configs retain third-party backbone settings."""
import copy
import json
import unittest
from unittest.mock import patch

from transformers import PretrainedConfig

from hyprorec.configuration_hocrs import HoCRSConfig


class LegacyOmniConfig(PretrainedConfig):
    model_type = "legacy_omni"

    def __init__(self, hidden_size=16, vocab_size=32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def to_dict(self):
        values = copy.deepcopy(self.__dict__)
        values["model_type"] = self.model_type
        return values


class BackboneConfigRoundtripTest(unittest.TestCase):
    def test_legacy_nested_config_can_be_serialized(self):
        backbone = LegacyOmniConfig(hidden_size=48, vocab_size=99)
        config = HoCRSConfig(backbone_config=backbone)
        saved = json.loads(config.to_json_string())
        self.assertEqual(saved["backbone_config"]["hidden_size"], 48)
        self.assertEqual(saved["backbone_config"]["vocab_size"], 99)
        self.assertIn("output_attentions", saved["backbone_config"])

    def test_custom_config_roundtrip_keeps_checkpoint_values(self):
        backbone = LegacyOmniConfig(
            hidden_size=48,
            vocab_size=99,
            auto_map={"AutoConfig": "configuration_legacy.LegacyOmniConfig"},
            _name_or_path="/saved/backbone",
        )
        config = HoCRSConfig(backbone_config=backbone)
        payload = json.loads(config.to_json_string(use_diff=False))
        with patch(
            "hyprorec.configuration_hocrs.AutoConfig.from_pretrained",
            return_value=LegacyOmniConfig(),
        ):
            restored = HoCRSConfig.from_dict(payload)
        self.assertIsInstance(restored.backbone_config, LegacyOmniConfig)
        self.assertEqual(restored.backbone_config.hidden_size, 48)
        self.assertEqual(restored.backbone_config.vocab_size, 99)


if __name__ == "__main__":
    unittest.main()
