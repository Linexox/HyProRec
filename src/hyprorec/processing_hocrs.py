"""Prompt serialization for the two independent HyProRec tasks."""

from __future__ import annotations

from collections.abc import Mapping

from transformers import ProcessorMixin

from .configuration_hocrs import ALL_GRAPH_VIEWS

NODE_START_TOKEN = "<|node_start|>"
NODE_TOKEN = "<|node|>"
NODE_END_TOKEN = "<|node_end|>"
HYPEREDGE_START_TOKEN = "<|hyperedge_start|>"
HYPEREDGE_TOKEN = "<|hyperedge|>"
HYPEREDGE_END_TOKEN = "<|hyperedge_end|>"
SOFT_PROMPT_TOKEN = "<|soft_prompt|>"


def graph_start_token(view: str) -> str:
    return f"<|{view}_hypergraph_start|>"


def graph_end_token(view: str) -> str:
    return f"<|{view}_hypergraph_end|>"


class HoCRSProcessor(ProcessorMixin):
    attributes = ["tokenizer"]
    tokenizer_class = ("PreTrainedTokenizerBase", "PreTrainedTokenizerFast")

    def __init__(
        self, tokenizer, num_prompt_tokens: int = 20, chat_template: str | None = None
    ) -> None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": self.get_special_tokens()}
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.num_prompt_tokens = num_prompt_tokens
        super().__init__(tokenizer=tokenizer, chat_template=chat_template)

    @staticmethod
    def get_special_tokens() -> list[str]:
        tokens = [
            NODE_START_TOKEN,
            NODE_TOKEN,
            NODE_END_TOKEN,
            HYPEREDGE_START_TOKEN,
            HYPEREDGE_TOKEN,
            HYPEREDGE_END_TOKEN,
            SOFT_PROMPT_TOKEN,
        ]
        for view in ALL_GRAPH_VIEWS:
            tokens.extend((graph_start_token(view), graph_end_token(view)))
        return tokens

    def get_token_id_map(self) -> dict[str, object]:
        convert = self.tokenizer.convert_tokens_to_ids
        return {
            "node_token_id": convert(NODE_TOKEN),
            "hyperedge_token_id": convert(HYPEREDGE_TOKEN),
            "soft_prompt_token_id": convert(SOFT_PROMPT_TOKEN),
            "graph_start_token_ids": {
                view: convert(graph_start_token(view)) for view in ALL_GRAPH_VIEWS
            },
            "graph_end_token_ids": {
                view: convert(graph_end_token(view)) for view in ALL_GRAPH_VIEWS
            },
        }

    def serialize_graph(self, view: str, num_nodes: int, num_hyperedges: int) -> str:
        return "".join(
            (
                graph_start_token(view),
                NODE_START_TOKEN,
                NODE_TOKEN * num_nodes,
                NODE_END_TOKEN,
                HYPEREDGE_START_TOKEN,
                HYPEREDGE_TOKEN * num_hyperedges,
                HYPEREDGE_END_TOKEN,
                graph_end_token(view),
            )
        )

    def build_prompt(
        self, context: str, graph_sizes: Mapping[str, tuple[int, int]]
    ) -> str:
        prompts = SOFT_PROMPT_TOKEN * self.num_prompt_tokens
        graphs = "\n".join(
            self.serialize_graph(view, *graph_sizes[view]) for view in graph_sizes
        )
        return f"{prompts}\n{context}\n{graphs}" if graphs else f"{prompts}\n{context}"

    def __call__(self, text, **kwargs):
        return self.tokenizer(text, **kwargs)


__all__ = [
    "HYPEREDGE_TOKEN",
    "NODE_TOKEN",
    "SOFT_PROMPT_TOKEN",
    "HoCRSProcessor",
    "graph_end_token",
    "graph_start_token",
]
