"""INT8 stays confined to the frozen backbone across config reloads."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import Qwen2Config

from hyprorec.arguments import ModelArguments
from hyprorec.backbones.base import QwenOmniAdapter, load_backbone_adapter
from hyprorec.configuration_hocrs import HoCRSConfig


class Int8BackboneTest(unittest.TestCase):
    def test_int8_requires_frozen_backbone(self):
        with self.assertRaisesRegex(ValueError, 'frozen'):
            ModelArguments(backbone_load_in_8bit=True, freeze_backbone=False)

    def test_loader_selects_explicit_int8_omni_path(self):
        with patch('hyprorec.backbones.base.AutoConfig.from_pretrained',
                   return_value=SimpleNamespace(model_type='qwen2_5_omni')):
            with patch.object(QwenOmniAdapter, 'from_pretrained') as load:
                load_backbone_adapter('local-omni', load_in_8bit=True)
        load.assert_called_once_with('local-omni', load_in_8bit=True)

    def test_other_backbones_do_not_silently_ignore_int8(self):
        with patch('hyprorec.backbones.base.AutoConfig.from_pretrained',
                   return_value=SimpleNamespace(model_type='qwen2')):
            with self.assertRaisesRegex(ValueError, 'Omni'):
                load_backbone_adapter('local-text', load_in_8bit=True)

    def test_reload_quantizer_excludes_all_non_backbone_modules(self):
        backbone = Qwen2Config()
        backbone.quantization_config = {
            'quant_method': 'bitsandbytes', 'load_in_8bit': True,
            'llm_int8_skip_modules': ['lm_head'],
        }
        config = HoCRSConfig(backbone_config=backbone)
        restored = HoCRSConfig.from_dict(config.to_dict())
        skip = restored.quantization_config['llm_int8_skip_modules']
        for module in ('lm_head', 'soft_prompt_embeddings', 'hypergraph_encoders',
                       'hypergraph_projectors', 'graph_moe', 'recommendation_head'):
            self.assertIn(module, skip)
        self.assertEqual(backbone.quantization_config['llm_int8_skip_modules'], ['lm_head'])


if __name__ == '__main__':
    unittest.main()
