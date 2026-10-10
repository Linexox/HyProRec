import unittest

import torch
from transformers import AutoModelForCausalLM, GPT2Config

from hyprorec.configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from hyprorec.modeling_hocrs import HoCRSRecommendationModel


class BackboneEncodingTest(unittest.TestCase):
    def test_encoding_keeps_only_last_hidden_state_by_default(self) -> None:
        backbone_config = GPT2Config(
            n_layer=2,
            n_head=2,
            n_embd=16,
            vocab_size=32,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
        )
        backbone = AutoModelForCausalLM.from_config(backbone_config)
        graph_config = HoCRSHypergraphConfig(
            input_dim=4,
            hidden_dim=4,
            output_dim=2,
            num_layers=1,
        )
        config = HoCRSConfig(
            backbone_config=backbone.config,
            views=("txt",),
            num_items=3,
            item_feature_dims={"txt": 4},
            item_table_view="txt",
            txt_hypergraph_config=graph_config,
            prompt_token_id=3,
            freeze_backbone=False,
        )
        model = HoCRSRecommendationModel(config, backbone=backbone).eval()
        model.initialize_feature_tables({"txt": torch.randn(3, 4)})

        input_ids = torch.randint(0, 3, (1, 24))
        input_ids[:, :20] = 3
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :20] = -100
        output = model.encode(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        with torch.no_grad():
            embeds = model._inject(input_ids, {})
            reference = backbone(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        self.assertEqual(len(output.hidden_states), 1)
        self.assertTrue(
            torch.allclose(output.hidden_states[0], reference.hidden_states[-1])
        )
        self.assertIsNone(output.logits)
        self.assertTrue(torch.allclose(output.loss, reference.loss))
        expected = reference.logits[:, :-1].argmax(dim=-1)
        expected[labels[:, 1:] == -100] = -100
        self.assertTrue(torch.equal(output.conversation_predictions, expected))

        generated = model.generate(
            input_ids=input_ids[:, :21],
            attention_mask=attention_mask[:, :21],
            hypergraphs={},
            max_new_tokens=1,
        )
        self.assertEqual(generated.shape, (1, 22))


if __name__ == "__main__":
    unittest.main()
