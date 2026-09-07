# START: Exercise the actual small Thinker architecture without downloading weights.
import tempfile
import unittest
from unittest import mock

import torch
from transformers import Qwen2_5OmniThinkerConfig

from hyprorec.configuration_hocrs import HoCRSConfig
from hyprorec.modeling_hocrs import HoCRSModel, load_backbone


class QwenBackboneTest(unittest.TestCase):
    def test_thinker_forward_backward_and_roundtrip(self):
        thinker = Qwen2_5OmniThinkerConfig(
            vision_start_token_id=20, vision_end_token_id=21,
            audio_start_token_id=22, audio_end_token_id=23,
            image_token_id=24, video_token_id=25, audio_token_id=26,
            text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, rope_parameters={
                                 "rope_type": "default", "rope_theta": 10000.0,
                                 "mrope_section": [1, 1, 2]}),
            audio_config=dict(d_model=16, encoder_layers=1,
                              encoder_attention_heads=2, encoder_ffn_dim=32,
                              output_dim=16),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                               num_heads=2, out_hidden_size=16),
        )
        config = HoCRSConfig(backbone_config=thinker, views=[], num_items=3,
                            item_dim=8, recommendation_hidden_dim=8,
                            num_soft_prompt_tokens=1, soft_prompt_token_id=3,
                            rec_token_id=4, freeze_backbone=True)
        model = HoCRSModel(config)
        batch = dict(input_ids=torch.tensor([[2, 3, 4, 5, 6]]),
                     attention_mask=torch.ones(1, 5, dtype=torch.long),
                     labels=torch.tensor([[-100, -100, -100, 5, 6]]),
                     rec_labels=torch.tensor([1]), use_cache=False)
        model.eval()
        output = model(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertGreater(model.soft_prompt_embeddings.weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = HoCRSModel.from_pretrained(directory).eval()
            self.assertEqual(restored.backbone.config.model_type, "qwen2_5_omni_thinker")
            torch.testing.assert_close(restored(**batch).rec_scores, output.rec_scores)

    def test_omni_loader_selects_thinker(self):
        with mock.patch("hyprorec.modeling_hocrs.AutoConfig.from_pretrained") as config, \
             mock.patch("transformers.Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained") as load:
            config.return_value.model_type = "qwen2_5_omni"
            self.assertIs(load_backbone("local-omni"), load.return_value)
            load.assert_called_once_with("local-omni", dtype="auto")
# END: Exercise the actual small Thinker architecture without downloading weights.
