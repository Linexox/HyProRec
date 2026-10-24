"""INT8 stays confined to the frozen backbone across config reloads."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import PreTrainedModel, Qwen2Config

from hyprorec.arguments import ModelArguments
from hyprorec.backbones.base import QwenOmniAdapter, load_backbone_adapter
from hyprorec.configuration_hocrs import HoCRSConfig
from hyprorec.modeling_hocrs import HoCRSRecommendationModel


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

    def test_reload_preserves_frozen_towers_and_trainable_prompts(self):
        model = SimpleNamespace(
            config=SimpleNamespace(backbone_config=SimpleNamespace(quantization_config={
                'load_in_8bit': True}), grounding_checkpoint_path='grounding'),
            backbone=torch.nn.Linear(2, 2),
            hypergraph_encoders=torch.nn.ModuleList([torch.nn.Linear(2, 2)]),
            soft_prompt_embeddings=torch.nn.Embedding(20, 2),
        )
        with patch.object(PreTrainedModel, 'from_pretrained', return_value=model):
            restored = HoCRSRecommendationModel.from_pretrained('checkpoint')
        self.assertFalse(restored.is_quantized)
        self.assertFalse(any(p.requires_grad for p in restored.backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in restored.hypergraph_encoders.parameters()))
        self.assertTrue(restored.soft_prompt_embeddings.weight.requires_grad)

    def test_state_reload_restores_runtime_quantization_scales(self):
        class RuntimeInt8Linear(torch.nn.Linear):
            def __init__(self):
                super().__init__(2, 2)
                self.weight.SCB = None
                self.state = SimpleNamespace(SCB=torch.ones(2))

            def _load_from_state_dict(self, *args, **kwargs):
                if self.weight.SCB is None:
                    raise RuntimeError('quantization scales unavailable on weight')
                return super()._load_from_state_dict(*args, **kwargs)

        config = HoCRSConfig(
            backbone_config=Qwen2Config(vocab_size=16, hidden_size=8,
                intermediate_size=16, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2),
            views=('txt',), num_items=3, item_feature_dims={'txt': 4},
            txt_hypergraph_config={'input_dim': 4, 'hidden_dim': 4,
                                  'output_dim': 4, 'num_layers': 1},
            recommendation_hidden_dim=8, recommendation_attention_dim=4,
        )
        model = HoCRSRecommendationModel(config)
        model.backbone = RuntimeInt8Linear()
        model.config.backbone_config.quantization_config = {'load_in_8bit': True}
        with patch('bitsandbytes.nn.Linear8bitLt', RuntimeInt8Linear):
            model.load_state_dict(model.state_dict())
        self.assertIs(model.backbone.weight.SCB, model.backbone.state.SCB)


if __name__ == '__main__':
    unittest.main()
