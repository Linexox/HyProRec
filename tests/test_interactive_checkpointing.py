"""InteractiveOmni wrapper checkpointing preserves frozen-decoder input gradients."""
import unittest

import torch
from transformers import PreTrainedModel, Qwen3Config, Qwen3ForCausalLM

from hyprorec.backbones.base import InteractiveOmniAdapter


class OmniWrapper(PreTrainedModel):
    config_class = Qwen3Config

    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen3ForCausalLM(config)


class InteractiveCheckpointingTest(unittest.TestCase):
    def test_checkpointing_targets_text_model_and_preserves_input_gradients(self):
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        model = OmniWrapper(config).requires_grad_(False)
        adapter = InteractiveOmniAdapter(model)
        with self.assertRaises(ValueError):
            model.gradient_checkpointing_enable()

        adapter.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.assertTrue(model.is_gradient_checkpointing)
        hidden = torch.randn(1, 4, 16, requires_grad=True)
        model.train()
        output = model.language_model(inputs_embeds=hidden, use_cache=False)
        output.logits.square().mean().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(hidden.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

        adapter.gradient_checkpointing_disable()
        self.assertFalse(model.is_gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
