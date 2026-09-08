import tempfile
import unittest
from unittest import mock

import torch
from transformers import (
    AutoModel,
    MPNetConfig,
    MPNetModel,
    VideoMAEConfig,
    VideoMAEModel,
    ViTConfig,
    ViTModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
)

from hyprorec.configuration_alignment import HoCRSAlignmentConfig
from hyprorec.modeling_alignment import HoCRSAlignmentModel


class AlignmentModelTest(unittest.TestCase):
    def test_all_pair_alignment_updates_source_encoders_and_heads(self) -> None:
        txt_model = MPNetModel(
            MPNetConfig(
                vocab_size=16,
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
            )
        )
        with torch.no_grad():
            txt_model.embeddings.word_embeddings.weight.fill_(0.125)
        img_model = ViTModel(
            ViTConfig(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                image_size=8,
                patch_size=4,
            )
        )
        ado_model = Wav2Vec2Model(
            Wav2Vec2Config(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                conv_dim=[8, 8],
                conv_stride=[2, 2],
                conv_kernel=[3, 3],
                num_conv_pos_embedding_groups=2,
            )
        )
        vdo_model = VideoMAEModel(
            VideoMAEConfig(
                hidden_size=8,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                image_size=8,
                patch_size=4,
                num_frames=4,
                tubelet_size=2,
            )
        )
        config = HoCRSAlignmentConfig(alignment_dim=4)
        with (
            mock.patch.object(MPNetModel, "from_pretrained", return_value=txt_model),
            mock.patch.object(ViTModel, "from_pretrained", return_value=img_model),
            mock.patch.object(Wav2Vec2Model, "from_pretrained", return_value=ado_model),
            mock.patch(
                "hyprorec.modeling_alignment._load_video_model",
                return_value=vdo_model,
            ),
        ):
            model = HoCRSAlignmentModel(config)

        self.assertAlmostEqual(
            float(
                model.source_encoders["txt"].embeddings.word_embeddings.weight.mean()
            ),
            0.125,
            places=6,
        )

        output = model(
            item_ids=torch.tensor([0, 1]),
            modality_mask=torch.ones(2, 4, dtype=torch.bool),
            txt_input_ids=torch.randint(0, 16, (2, 4)),
            txt_attention_mask=torch.ones(2, 4, dtype=torch.long),
            img_pixel_values=torch.randn(2, 3, 8, 8),
            ado_input_values=torch.randn(2, 64),
            vdo_pixel_values=torch.randn(2, 4, 3, 8, 8),
        )

        self.assertEqual(set(output.embeddings), {"txt", "img", "ado", "vdo"})
        self.assertGreater(float(output.loss), 0.0)
        output.loss.backward()
        self.assertIsNotNone(model.alignment_heads["txt"].weight.grad)
        self.assertIsNotNone(
            model.source_encoders["txt"].embeddings.word_embeddings.weight.grad
        )

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            with (
                mock.patch.object(
                    MPNetModel,
                    "from_pretrained",
                    side_effect=lambda *_: MPNetModel(txt_model.config),
                ),
                mock.patch.object(
                    ViTModel,
                    "from_pretrained",
                    side_effect=lambda *_: ViTModel(img_model.config),
                ),
                mock.patch.object(
                    Wav2Vec2Model,
                    "from_pretrained",
                    side_effect=lambda *_: Wav2Vec2Model(ado_model.config),
                ),
                mock.patch(
                    "hyprorec.modeling_alignment._load_video_model",
                    side_effect=lambda *_: VideoMAEModel(vdo_model.config),
                ),
            ):
                reloaded = HoCRSAlignmentModel.from_pretrained(directory)
                auto_model = AutoModel.from_pretrained(directory)

        self.assertIsInstance(auto_model, HoCRSAlignmentModel)
        self.assertTrue(
            torch.equal(
                model.alignment_heads["txt"].weight,
                reloaded.alignment_heads["txt"].weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
