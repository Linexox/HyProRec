"""Stable interfaces for the language side of supported multimodal backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


@dataclass
class BackboneOutput:
    """Decoder output fields consumed by the CRS model."""

    raw: Any
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None
    past_key_values: Any
    extra_cache: dict[str, Any]


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    value = getattr(config, name, default)
    if value is not None:
        return value
    if isinstance(config, dict):
        return config.get(name, default)
    return default


def _hidden_size(config: Any) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        value = _config_value(config, name)
        if value is not None:
            return int(value)
    for name in ("text_config", "llm_config", "thinker_config"):
        nested = _config_value(config, name)
        if nested is not None:
            size = _hidden_size(nested)
            if size:
                return size
    raise AttributeError("backbone config has no hidden size")


class BackboneAdapter:
    """Adapter contract shared by every CRS backbone.

    The adapter deliberately exposes only the text decoder path.  Native
    vision/audio/video inputs are outside the HyProRec CRS data flow; graph
    tokens are injected through ``inputs_embeds`` instead.
    """

    model_type: str = "generic"

    def __init__(
        self,
        model: nn.Module,
        config: Any | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        self.model = model
        self.config = config or model.config
        self.tokenizer = tokenizer

    @classmethod
    def load_tokenizer(cls, name_or_path: str) -> PreTrainedTokenizerBase:
        try:
            return AutoTokenizer.from_pretrained(name_or_path)
        except ValueError:
            return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

    @classmethod
    def load_model(cls, name_or_path: str) -> nn.Module:
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "BackboneAdapter":
        tokenizer = cls.load_tokenizer(name_or_path)
        model = cls.load_model(name_or_path)
        return cls(model=model, config=model.config, tokenizer=tokenizer)

    @classmethod
    def from_model(cls, model: nn.Module) -> "BackboneAdapter":
        return cls(model=model, config=model.config)

    @classmethod
    def from_config(cls, config: Any) -> "BackboneAdapter":
        return cls(model=cls._model_from_config(config), config=config)

    @property
    def hidden_size(self) -> int:
        return _hidden_size(self.config)

    def get_input_embeddings(self) -> nn.Module:
        embeddings = self.model.get_input_embeddings()
        if embeddings is not None:
            return embeddings
        return self._decoder().get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        setter = getattr(self.model, "set_input_embeddings", None)
        if setter is not None:
            setter(value)
        else:
            self._decoder().set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        getter = getattr(self.model, "get_output_embeddings", None)
        embeddings = getter() if getter is not None else None
        if embeddings is not None:
            return embeddings
        decoder = self._decoder()
        embeddings = decoder.get_output_embeddings()
        if embeddings is not None:
            return embeddings
        if hasattr(decoder, "lm_head"):
            return decoder.lm_head
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        raise AttributeError("backbone has no language-model output embeddings")

    def set_output_embeddings(self, value: nn.Module) -> None:
        setter = getattr(self.model, "set_output_embeddings", None)
        if setter is not None:
            setter(value)
            return
        decoder = self._decoder()
        setter = getattr(decoder, "set_output_embeddings", None)
        if setter is not None:
            setter(value)
            return
        if hasattr(decoder, "lm_head"):
            decoder.lm_head = value
            return
        if hasattr(self.model, "lm_head"):
            self.model.lm_head = value
            return
        raise AttributeError("backbone has no language-model output embeddings")

    def resize_token_embeddings(self, size: int) -> nn.Module:
        resize = getattr(self.model, "resize_token_embeddings", None)
        if resize is not None:
            return resize(size)
        return self._decoder().resize_token_embeddings(size)

    def encode_condition(
        self,
        *,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        past_key_values: Any = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> BackboneOutput:
        decoder_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=use_cache,
            **kwargs,
        )
        if past_key_values is not None:
            decoder_kwargs["past_key_values"] = past_key_values
        output = self._decoder()(**decoder_kwargs)
        return self._output(output)

    def get_last_hidden_state(self, output: BackboneOutput) -> torch.Tensor:
        return output.last_hidden_state

    def compute_lm_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.get_output_embeddings()(hidden_states)

    def prepare_generation_cache(self, output: BackboneOutput) -> dict[str, Any]:
        return {
            "past_key_values": output.past_key_values,
            **output.extra_cache,
        }

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing_disable()

    def _output(self, output: Any) -> BackboneOutput:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = getattr(output, "hidden_states", None)[-1]
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is not None:
            hidden_states = tuple(hidden_states)
        extra_cache = {
            name: getattr(output, name)
            for name in ("rope_deltas", "cache_position")
            if getattr(output, name, None) is not None
        }
        return BackboneOutput(
            raw=output,
            last_hidden_state=hidden,
            hidden_states=hidden_states,
            past_key_values=getattr(output, "past_key_values", None),
            extra_cache=extra_cache,
        )

    def _decoder(self) -> nn.Module:
        for name in ("model", "transformer", "language_model", "llm"):
            decoder = getattr(self.model, name, None)
            if not isinstance(decoder, nn.Module):
                continue
            nested = getattr(decoder, "model", None)
            return nested if isinstance(nested, nn.Module) else decoder
        return self.model

    @classmethod
    def _model_from_config(cls, config: Any) -> nn.Module:
        return AutoModelForCausalLM.from_config(config)


class QwenOmniAdapter(BackboneAdapter):
    model_type = "qwen2_5_omni"

    @classmethod
    def load_model(cls, name_or_path: str) -> nn.Module:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

    @classmethod
    def _model_from_config(cls, config: Any) -> nn.Module:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        thinker_config = _config_value(config, "thinker_config") or config
        return Qwen2_5OmniThinkerForConditionalGeneration(thinker_config)


class InteractiveOmniAdapter(BackboneAdapter):
    model_type = "interactiveomni"

    @classmethod
    def load_model(cls, name_or_path: str) -> nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    @classmethod
    def _model_from_config(cls, config: Any) -> nn.Module:
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


class QwenVLAdapter(BackboneAdapter):
    model_type = "qwen2_5_vl"

    @classmethod
    def load_model(cls, name_or_path: str) -> nn.Module:
        return AutoModelForImageTextToText.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

    @classmethod
    def _model_from_config(cls, config: Any) -> nn.Module:
        return AutoModelForImageTextToText.from_config(config)


class QwenTextAdapter(BackboneAdapter):
    model_type = "qwen2"

    @classmethod
    def load_model(cls, name_or_path: str) -> nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )


class LlavaAdapter(BackboneAdapter):
    model_type = "llava"

    @classmethod
    def load_model(cls, name_or_path: str) -> nn.Module:
        return AutoModelForImageTextToText.from_pretrained(
            name_or_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

    @classmethod
    def _model_from_config(cls, config: Any) -> nn.Module:
        return AutoModelForImageTextToText.from_config(config)


ADAPTERS = {
    adapter.model_type: adapter
    for adapter in (
        QwenOmniAdapter,
        InteractiveOmniAdapter,
        QwenVLAdapter,
        QwenTextAdapter,
        LlavaAdapter,
    )
}


def load_backbone_adapter(name_or_path: str) -> BackboneAdapter:
    try:
        config = AutoConfig.from_pretrained(name_or_path)
    except ValueError:
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
    adapter_type = ADAPTERS.get(config.model_type, BackboneAdapter)
    return adapter_type.from_pretrained(name_or_path)


def adapter_from_model(model: nn.Module) -> BackboneAdapter:
    adapter_type = ADAPTERS.get(getattr(model.config, "model_type", ""), BackboneAdapter)
    return adapter_type.from_model(model)


def adapter_from_config(config: Any) -> BackboneAdapter:
    adapter_type = ADAPTERS.get(_config_value(config, "model_type", ""), BackboneAdapter)
    return adapter_type.from_config(config)


__all__ = [
    "ADAPTERS",
    "BackboneAdapter",
    "BackboneOutput",
    "adapter_from_config",
    "adapter_from_model",
    "load_backbone_adapter",
]
