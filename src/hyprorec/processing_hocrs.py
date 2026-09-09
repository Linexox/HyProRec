"""Tokenizer processor and graph-prompt serialization for HoCRS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from transformers import ProcessorMixin

from .constants import GRAPH_VIEWS

NODE_START_TOKEN = "<|node_start|>"
NODE_TOKEN = "<|node|>"
NODE_END_TOKEN = "<|node_end|>"
HYPEREDGE_START_TOKEN = "<|hyperedge_start|>"
HYPEREDGE_TOKEN = "<|hyperedge|>"
HYPEREDGE_END_TOKEN = "<|hyperedge_end|>"
REC_TOKEN = "<|rec|>"
SOFT_PROMPT_TOKEN = "<|soft_prompt|>"


def graph_start_token(view: str) -> str:
    return f"<|{view}_hypergraph_start|>"


def graph_end_token(view: str) -> str:
    return f"<|{view}_hypergraph_end|>"


class HoCRSProcessor(ProcessorMixin):
    """A tokenizer-only processor that serializes variable-size graph blocks."""

    # START: Declare the single processor component required by Transformers.
    attributes = ["tokenizer"]
    tokenizer_class = ("PreTrainedTokenizerBase", "PreTrainedTokenizerFast")
    # END: Declare the single processor component required by Transformers.

    def __init__(
        self,
        tokenizer,
        num_soft_prompt_tokens: int = 10,
        chat_template: str | None = None,
    ) -> None:
        if num_soft_prompt_tokens < 0:
            raise ValueError("num_soft_prompt_tokens must be non-negative.")
        tokenizer.add_special_tokens(
            {"additional_special_tokens": self.get_special_tokens()}
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("The tokenizer needs an EOS or padding token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        self.num_soft_prompt_tokens = num_soft_prompt_tokens
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
            REC_TOKEN,
            SOFT_PROMPT_TOKEN,
        ]
        for view in GRAPH_VIEWS:
            tokens.extend((graph_start_token(view), graph_end_token(view)))
        return tokens

    def get_token_id_map(self) -> dict[str, Any]:
        convert = self.tokenizer.convert_tokens_to_ids
        boundary_tokens = [
            NODE_START_TOKEN,
            NODE_END_TOKEN,
            HYPEREDGE_START_TOKEN,
            HYPEREDGE_END_TOKEN,
            REC_TOKEN,
            *[graph_start_token(view) for view in GRAPH_VIEWS],
            *[graph_end_token(view) for view in GRAPH_VIEWS],
        ]
        return {
            "node_token_id": convert(NODE_TOKEN),
            "hyperedge_token_id": convert(HYPEREDGE_TOKEN),
            "rec_token_id": convert(REC_TOKEN),
            "soft_prompt_token_id": convert(SOFT_PROMPT_TOKEN),
            "graph_start_token_ids": {
                view: convert(graph_start_token(view)) for view in GRAPH_VIEWS
            },
            "graph_end_token_ids": {
                view: convert(graph_end_token(view)) for view in GRAPH_VIEWS
            },
            "trainable_special_token_ids": [
                convert(token) for token in boundary_tokens
            ],
        }

    def serialize_graph(self, view: str, num_nodes: int, num_hyperedges: int) -> str:
        if view not in GRAPH_VIEWS:
            raise ValueError(f"Unknown graph view: {view}")
        if num_nodes <= 0 or num_hyperedges <= 0:
            raise ValueError("Serialized hypergraphs must contain nodes and hyperedges.")

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
        self,
        context: str,
        graph_sizes: Mapping[str, tuple[int, int]],
    ) -> str:
        graph_text = "\n".join(
            f"{view.upper()} hypergraph: "
            f"{self.serialize_graph(view, *graph_sizes[view])}"
            for view in graph_sizes
        )
        soft_prompts = SOFT_PROMPT_TOKEN * self.num_soft_prompt_tokens

        if graph_text:
            user_text = (
                "Use the conversation history and the supplied hypergraphs to infer "
                "the user's preference, recommend an appropriate movie, and respond.\n"
                f"Conversation History:\n{context}\n"
                f"Hypergraphs:\n{graph_text}\n"
                f"Recommendation state: {REC_TOKEN}{soft_prompts}"
            )
        else:
            user_text = (
                "Use the conversation history to infer the user's preference, "
                "recommend an appropriate movie, and respond.\n"
                f"Conversation History:\n{context}\n"
                f"Recommendation state: {REC_TOKEN}{soft_prompts}"
            )
        
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"User: {user_text}\nAssistant:"

    def __call__(self, text: str | Sequence[str], **kwargs: Any):
        return self.tokenizer(text, **kwargs)


__all__ = [
    "HYPEREDGE_TOKEN",
    "NODE_TOKEN",
    "REC_TOKEN",
    "SOFT_PROMPT_TOKEN",
    "HoCRSProcessor",
    "graph_end_token",
    "graph_start_token",
]
