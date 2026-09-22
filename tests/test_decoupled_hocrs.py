import json
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    PreTrainedTokenizerFast,
    Qwen2_5OmniThinkerConfig,
    Trainer,
    TrainingArguments,
)

from hyprorec.configuration_hocrs import HoCRSConfig, HoCRSHypergraphConfig
from hyprorec.data import HoCRSDataCollator, HoCRSDataset, HoCRSDatasetConfig
from hyprorec.data.hypergraph import HypergraphTable
from hyprorec.data.grounding import HoCRSGroundingDataset
from hyprorec.configuration_grounding import HoCRSGroundingConfig
from hyprorec.modeling_hocrs import HoCRSConversationModel, HoCRSRecommendationModel
from hyprorec.metrics import build_compute_metrics, preprocess_logits_for_metrics
from hyprorec.processing_hocrs import HoCRSProcessor
from hyprorec.scripts.prepare_hyperedge_table import prepare_hyperedge_table


def processor():
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[EOS]": 1,
                "Seeker": 2,
                "Recommender": 3,
                "hello": 4,
                "film": 5,
                "reply": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]"
    )
    return HoCRSProcessor(fast)


def config(
    processor, task="recommendation", head="item_table", views=("txt",), item_view="txt"
):
    graph = HoCRSHypergraphConfig(
        input_dim=8, hidden_dim=16, output_dim=8, num_layers=2
    )
    return HoCRSConfig(
        backbone_config=GPT2Config(
            vocab_size=len(processor.tokenizer),
            n_embd=32,
            n_layer=1,
            n_head=4,
            n_positions=256,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        ),
        task=task,
        recommendation_head=head,
        views=views,
        txt_hypergraph_config=graph,
        co_hypergraph_config=graph,
        item_feature_dims={"txt": 8, "img": 8},
        item_table_view=item_view,
        num_items=12,
        prompt_token_id=processor.get_token_id_map()["soft_prompt_token_id"],
        recommendation_hidden_dim=32,
    )


def fixture(directory):
    root = Path(directory)
    conversation = [
        {
            "dialog": [
                {"role": "Seeker", "text": "hello", "items": [1]},
                {"role": "Recommender", "text": "film", "items": [2, 3]},
                {"role": "Seeker", "text": "reply", "items": []},
            ]
        }
    ]
    for split in ("train", "valid", "test"):
        (root / f"{split}_data.json").write_text(
            json.dumps(conversation), encoding="utf-8"
        )
    rows = [[item, (item + 1) % 12, (item + 2) % 12] for item in range(12)]
    (root / "hyperedge_table.json").write_text(
        json.dumps({"txt": rows, "co": rows}), encoding="utf-8"
    )
    return root


def test_turn_tasks_and_input_order():
    with tempfile.TemporaryDirectory() as directory:
        root = fixture(directory)
        rec = HoCRSDataset(
            HoCRSDatasetConfig(
                str(root), task="recommendation", views=("txt",), sample_repeat=2
            ),
            "train",
        )
        conv = HoCRSDataset(
            HoCRSDatasetConfig(str(root), task="conversation", views=("txt",)), "train"
        )
        assert len(rec) == 6
        assert len(conv) == 3
        assert rec[0]["context_item_ids"] == []
        assert rec[2]["context_item_ids"] == [1]
        p = processor()
        batch = HoCRSDataCollator(p, "recommendation")([rec[0], rec[2]])
        assert batch["rec_labels"].tolist() == [1, 2]
        assert set(batch["hypergraphs"]["txt"]["node_positions"][:, 0].tolist()) == {1}
        prompt_id = p.get_token_id_map()["soft_prompt_token_id"]
        assert (batch["input_ids"][:, :20] == prompt_id).all()
        assert batch["hypergraphs"]["txt"]["node_positions"][:, 1].min() > 20
        dialogue = HoCRSDataCollator(p, "conversation")([conv[0], conv[1]])
        assert (dialogue["labels"] != -100).any()
        assert (dialogue["labels"][:, :20] == -100).all()


def test_random_topk_varies_and_has_no_node_limit():
    table = HypergraphTable(
        {
            "txt": [
                [item, *((item + offset) % 100 for offset in range(1, 11))]
                for item in range(100)
            ]
        }
    )
    strict = table.build_local([0], "txt", 3, 1)
    random_graphs = [table.build_local([0], "txt", 3, 1, "random") for _ in range(10)]
    assert strict.node_ids.tolist() == [0, 1, 2, 3]
    assert any(
        graph.node_ids.tolist() != strict.node_ids.tolist() for graph in random_graphs
    )
    assert table.build_local(list(range(50)), "txt", 3, 1).num_hyperedges == 50


def test_recommendation_heads_and_conversation_are_independent():
    p = processor()
    input_ids = torch.tensor(
        [[p.get_token_id_map()["soft_prompt_token_id"]] * 20 + [2, 3, 4]]
    )
    attention = torch.ones_like(input_ids)
    features = {"txt": torch.randn(12, 8), "img": torch.randn(12, 8)}
    for head, view in (("item_table", "txt"), ("item_table", "full"), ("mlp", "txt")):
        model = HoCRSRecommendationModel(config(p, head=head, views=(), item_view=view))
        needed = features if view == "full" else {"txt": features["txt"]}
        if head == "mlp":
            needed = {}
        model.initialize_feature_tables(needed)
        output = model(
            input_ids,
            attention_mask=attention,
            pooling_mask=attention.bool(),
            rec_labels=torch.tensor([2]),
        )
        assert output.logits.shape == (1, 12)
        output.loss.backward()
        assert model.soft_prompt_embeddings.weight.grad.abs().sum() > 0
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = HoCRSRecommendationModel.from_pretrained(directory)
            assert restored.config.recommendation_head == head
            if head == "item_table":
                torch.testing.assert_close(restored.txt_feature_table, features["txt"])
    dialogue = HoCRSConversationModel(config(p, task="conversation", views=()))
    labels = input_ids.clone()
    labels[:, :20] = -100
    output = dialogue(input_ids, attention_mask=attention, labels=labels)
    assert torch.isfinite(output.loss)
    assert dialogue.generate(input_ids, max_new_tokens=2).shape[1] == 25


def test_single_co_view_uses_chosen_modality_features():
    p = processor()
    c = config(p, views=("co",))
    c.co_feature_view = "txt"
    model = HoCRSRecommendationModel(c)
    model.initialize_feature_tables(
        {"txt": torch.randn(12, 8), "img": torch.randn(12, 8)}
    )
    graph = {
        "node_ids": torch.tensor([1, 2]),
        "hyperedge_index": torch.tensor([[0, 1], [0, 0]]),
        "edge_ptr": torch.tensor([0, 1]),
        "node_positions": torch.tensor([[0, 21], [0, 22]]),
        "hyperedge_positions": torch.tensor([[0, 23]]),
    }
    prompt_id = p.get_token_id_map()["soft_prompt_token_id"]
    ids = torch.tensor([[prompt_id] * 20 + [2, 3, 4, 5]])
    output = model(
        ids,
        attention_mask=torch.ones_like(ids),
        pooling_mask=torch.ones_like(ids).bool(),
        rec_labels=torch.tensor([2]),
        hypergraphs={"co": graph},
    )
    output.loss.backward()
    assert model.hypergraph_projectors["co"].node_projector.weight.grad.abs().sum() > 0


def test_grounding_has_four_semantic_and_four_co_towers():
    modalities = ("txt", "img", "ado", "vdo")
    views = (*modalities, *(f"co_{view}" for view in modalities))
    config = HoCRSGroundingConfig(views=views, input_dims={view: 8 for view in views})
    assert len(config.views) == 8
    rows = [[item, (item + 1) % 12, (item + 2) % 12] for item in range(12)]
    table = HypergraphTable({**{view: rows for view in modalities}, "co": rows})
    sample = HoCRSGroundingDataset(table, views, topk=2, khop=1)[0]
    assert len(sample["hypergraphs"]) == 8
    assert all(sample["hypergraphs"][view].view == "co" for view in views[4:])


def test_trainer_evaluates_both_tasks():
    p = processor()
    with tempfile.TemporaryDirectory() as directory:
        root = fixture(directory)
        for task in ("recommendation", "conversation"):
            dataset = HoCRSDataset(
                HoCRSDatasetConfig(str(root), task=task, views=("txt",)), "validation"
            )
            model = (
                HoCRSRecommendationModel(config(p, views=("txt",)))
                if task == "recommendation"
                else HoCRSConversationModel(
                    config(p, task="conversation", views=("txt",))
                )
            )
            model.initialize_feature_tables({"txt": torch.randn(12, 8)})
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(root / task),
                    use_cpu=True,
                    report_to=[],
                    per_device_eval_batch_size=2,
                    remove_unused_columns=False,
                    label_names=(
                        ["rec_labels"] if task == "recommendation" else ["labels"]
                    ),
                ),
                eval_dataset=dataset,
                data_collator=HoCRSDataCollator(p, task),
                processing_class=p,
                compute_metrics=build_compute_metrics(p, task),
                preprocess_logits_for_metrics=preprocess_logits_for_metrics(task),
            )
            metrics = trainer.evaluate()
            assert (
                "eval_recall@50" in metrics
                if task == "recommendation"
                else "eval_bleu@1" in metrics
            )


def test_trainer_updates_each_task_prompt_independently():
    p = processor()
    with tempfile.TemporaryDirectory() as directory:
        root = fixture(directory)
        for task in ("recommendation", "conversation"):
            dataset = HoCRSDataset(
                HoCRSDatasetConfig(str(root), task=task, views=("txt",)), "train"
            )
            model = (
                HoCRSRecommendationModel(config(p, views=("txt",)))
                if task == "recommendation"
                else HoCRSConversationModel(
                    config(p, task="conversation", views=("txt",))
                )
            )
            model.initialize_feature_tables({"txt": torch.randn(12, 8)})
            before = model.soft_prompt_embeddings.weight.detach().clone()
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=str(root / f"training-{task}"),
                    use_cpu=True,
                    report_to=[],
                    per_device_train_batch_size=2,
                    max_steps=1,
                    remove_unused_columns=False,
                    label_names=(
                        ["rec_labels"] if task == "recommendation" else ["labels"]
                    ),
                ),
                train_dataset=dataset,
                data_collator=HoCRSDataCollator(p, task),
                processing_class=p,
            )
            assert torch.isfinite(torch.tensor(trainer.train().training_loss))
            assert not torch.equal(before, model.soft_prompt_embeddings.weight)


def test_top10_tables_and_train_only_cooccurrence():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        embeddings = root / "embeddings"
        embeddings.mkdir()
        for view in ("txt", "img", "ado", "vdo"):
            torch.save(torch.randn(12, 8), embeddings / f"{view}_embeddings.pt")
        train = [{"dialog": [{"role": "Seeker", "text": "a", "items": [1, 2]}]}]
        valid = [{"dialog": [{"role": "Seeker", "text": "b", "items": [1, 3]}]}]
        (root / "train_data.json").write_text(json.dumps(train), encoding="utf-8")
        (root / "valid_data.json").write_text(json.dumps(valid), encoding="utf-8")
        prepare_hyperedge_table(
            root, root / "hyperedge_table.json", topk=10, device=torch.device("cpu")
        )
        table = HypergraphTable.from_json(root / "hyperedge_table.json")
        assert set(table.tables) == {"txt", "img", "ado", "vdo", "co"}
        assert all(len(row) == 11 for row in table.tables["txt"])
        assert table.tables["co"][1] == [1, 2]
        assert 3 not in table.tables["co"][1]


def test_small_qwen_thinker_recommendation_path():
    p = processor()
    thinker = Qwen2_5OmniThinkerConfig(
        vision_start_token_id=20,
        vision_end_token_id=21,
        audio_start_token_id=22,
        audio_end_token_id=23,
        image_token_id=24,
        video_token_id=25,
        audio_token_id=26,
        text_config={
            "vocab_size": len(p.tokenizer),
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 2],
            },
        },
        audio_config={
            "d_model": 16,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "output_dim": 16,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
        },
    )
    c = HoCRSConfig(
        backbone_config=thinker,
        task="recommendation",
        views=(),
        item_feature_dims={"txt": 8},
        item_table_view="txt",
        num_items=12,
        prompt_token_id=p.get_token_id_map()["soft_prompt_token_id"],
        recommendation_hidden_dim=16,
    )
    model = HoCRSRecommendationModel(c)
    model.initialize_feature_tables({"txt": torch.randn(12, 8)})
    ids = torch.tensor([[c.prompt_token_id] * 20 + [2, 3]])
    output = model(
        ids,
        attention_mask=torch.ones_like(ids),
        pooling_mask=torch.ones_like(ids).bool(),
        rec_labels=torch.tensor([2]),
    )
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.soft_prompt_embeddings.weight.grad.abs().sum() > 0
    with tempfile.TemporaryDirectory() as directory:
        model.eval()
        model.save_pretrained(directory)
        restored = HoCRSRecommendationModel.from_pretrained(directory).eval()
        expected = model(
            ids,
            attention_mask=torch.ones_like(ids),
            pooling_mask=torch.ones_like(ids).bool(),
        ).logits
        actual = restored(
            ids,
            attention_mask=torch.ones_like(ids),
            pooling_mask=torch.ones_like(ids).bool(),
        ).logits
        torch.testing.assert_close(expected, actual)
    dialogue_config = HoCRSConfig(
        backbone_config=thinker,
        task="conversation",
        views=(),
        item_feature_dims={"txt": 8},
        num_items=12,
        prompt_token_id=c.prompt_token_id,
        recommendation_hidden_dim=16,
    )
    dialogue = HoCRSConversationModel(dialogue_config).eval()
    assert dialogue.generate(ids, max_new_tokens=2).size(1) == ids.size(1) + 2
