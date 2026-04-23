# -*- coding: utf-8 -*-
# Copyright (c) 2025 Meituan
# This code is licensed under the MIT License, for details, see the ./LICENSE file.

from typing import Optional, Tuple, Dict, List

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import auto_docstring
from transformers.models.longcat_flash.modeling_longcat_flash import (
    LongcatFlashForCausalLM,
    LongcatFlashModel,
    LongcatFlashRMSNorm,
    LongcatFlashRotaryEmbedding,
    LongcatFlashDecoderLayer,
    LongcatFlashPreTrainedModel,
)
from transformers.models.longcat_flash import LongcatFlashConfig

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig


class LongcatFlashNgramConfig(LongcatFlashConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongcatFlashNgramModel`]. It is used to instantiate
    a LongCat Flash model with N-gram enhanced embeddings according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the LongCat Flash model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`LongcatFlashNgramModel`]
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 56):
            Number of hidden layers in the Transformer decoder.
        num_layers (`int`, *optional*, defaults to 28):
            Number of layers, each with 2 sublayers.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting from a multi-head checkpoint to a GQA checkpoint, each group key and value head should be
            constructed by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        ffn_hidden_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            The rank of the query LoRA projection in MLA (Multi-head Latent Attention).
        kv_lora_rank (`int`, *optional*, defaults to 512):
            The rank of the key-value LoRA projection in MLA.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            The dimension of the non-position encoding part of query/key heads.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            The dimension of the RoPE part of query/key heads.
        head_dim (`int`, *optional*, defaults to 64):
            Standard dimension of qk heads, unused except for CI.
        v_head_dim (`int`, *optional*, defaults to 128):
            The dimension of value heads.
        qk_head_dim (`int`, *optional*):
            The total dimension of query/key heads. If not specified, set to `qk_nope_head_dim + qk_rope_head_dim`.
        moe_topk (`int`, *optional*, defaults to 12):
            Number of experts to route to for each token in the MoE layer.
        n_routed_experts (`int`, *optional*, defaults to 512):
            Number of routed experts in the MoE layer.
        zero_expert_num (`int`, *optional*, defaults to 256):
            Number of zero experts (identity function) to add to the expert pool.
        expert_ffn_hidden_size (`int`, *optional*, defaults to 2048):
            Hidden size of individual expert FFN layers.
        routed_scaling_factor (`float`, *optional*, defaults to 6.0):
            Scaling factor applied to the routing weights.
        emb_neighbor_num (`int`, *optional*):
            Maximum N-gram length for N-gram embeddings. This parameter determines the context window size for N-gram computation. Higher values capture
            longer-range lexical patterns but increase memory usage.
        emb_split_num (`int`, *optional*):
            Number of hash functions (or splits) to use for N-gram embeddings. Multiple hash functions help improve the quality of N-gram representations.
        ngram_vocab_size_ratio (`float`, *optional*):
            Ratio multiplier for N-gram vocabulary size relative to the base vocabulary size. The N-gram vocabulary
            size is calculated as `vocab_size * ngram_vocab_size_ratio`.
    Example:
    ```python
    >>> from transformers import LongcatFlashNgramModel, LongcatFlashNgramConfig
    >>> # Initializing a LongCat Flash N-gram style configuration
    >>> configuration = LongcatFlashNgramConfig(
    ...     emb_neighbor_num=3,
    ...     emb_split_num=4,
    ...     ngram_vocab_size_ratio=1.5
    ... )
    >>> # Initializing a model from the configuration
    >>> model = LongcatFlashNgramModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "longcat_flash_ngram"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.*.q_b_proj": "colwise",
        "layers.*.self_attn.*.kv_b_proj": "colwise",
        "layers.*.self_attn.*.o_proj": "rowwise",
        "layers.*.mlps.*.gate_proj": "colwise",
        "layers.*.mlps.*.up_proj": "colwise",
        "layers.*.mlps.*.down_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=6144,
        num_hidden_layers=56,
        num_layers=28,
        num_attention_heads=64,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        ffn_hidden_size=12288,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        head_dim=64,
        v_head_dim=128,
        qk_head_dim=None,
        moe_topk=12,
        n_routed_experts=512,
        zero_expert_num=256,
        expert_ffn_hidden_size=2048,
        routed_scaling_factor=6.0,
        emb_neighbor_num=None,
        emb_split_num=None,
        ngram_vocab_size_ratio=None,
        **kwargs,
    ):
        # N-gram embedding specific parameters
        self.emb_neighbor_num = emb_neighbor_num
        self.emb_split_num = emb_split_num
        self.ngram_vocab_size_ratio = ngram_vocab_size_ratio

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            ffn_hidden_size=ffn_hidden_size,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            qk_head_dim=qk_head_dim,
            moe_topk=moe_topk,
            n_routed_experts=n_routed_experts,
            zero_expert_num=zero_expert_num,
            expert_ffn_hidden_size=expert_ffn_hidden_size,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )


@auto_docstring
class LongcatFlashNgramPreTrainedModel(LongcatFlashPreTrainedModel):
    pass


class NgramCache(DynamicCache):
    """
    Extended DynamicCache for storing N-gram context alongside KV cache.
    """

    def __init__(self, config=None):
        super().__init__()
        self.ngram_context = None
        # Keep only n-1 tokens (minimum needed for N-gram computation)
        self.max_context_len = config.emb_neighbor_num - 1

    def update_ngram_context(self, new_tokens: torch.Tensor) -> None:
        """
        Update N-gram context with window management.

        Args:
            new_tokens: New tokens to append, shape (batch_size, seq_len)
        """
        if self.ngram_context is None:
            self.ngram_context = new_tokens.clone()
        else:
            self.ngram_context = torch.cat([self.ngram_context, new_tokens], dim=-1)

        # Truncate to maintain constant memory footprint
        if self.ngram_context.size(-1) > self.max_context_len:
            self.ngram_context = self.ngram_context[..., -self.max_context_len :]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "Cache":
        """Reorder cache for beam search."""
        # Reorder parent's KV cache
        super().reorder_cache(beam_idx)

        # Reorder N-gram context
        if self.ngram_context is not None:
            self.ngram_context = self.ngram_context.index_select(
                0, beam_idx.to(self.ngram_context.device)
            )

        return self


class NgramEmbedding(nn.Module):
    """
    Computes embeddings enriched with N-gram features without maintaining internal state.
    """

    def __init__(
        self,
        config,
        base_embeddings,
        ngram_vocab_size_ratio: int,
        vocab_size: int,
        emb_split_num: int,
        emb_neighbor_num: int,
        hidden_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.word_embeddings = base_embeddings

        self.vocab_size = vocab_size
        self.ngram_hash_modulus_base = ngram_vocab_size_ratio * vocab_size
        self.emb_split_num = emb_split_num
        self.emb_neighbor_num = emb_neighbor_num
        self.hidden_size = hidden_size

        self._init_ngram_embeddings(quant_config)
        self._vocab_mods_cache = None

    def _init_ngram_embeddings(
        self, quant_config: Optional[QuantizationConfig] = None, prefix: str = ""
    ) -> None:
        """Initialize N-gram embedding and projection layers."""
        num_embedders = self.emb_split_num * (self.emb_neighbor_num - 1)
        emb_dim = self.hidden_size // num_embedders

        embedders = []
        post_projs = []

        for i in range(num_embedders):
            vocab_size = self.ngram_hash_modulus_base + i * 2 + 1
            emb = nn.Embedding(vocab_size, emb_dim)
            proj = nn.Linear(emb_dim, self.hidden_size, bias=False)
            embedders.append(emb)
            post_projs.append(proj)

        self.embedders = nn.ModuleList(embedders)
        self.post_projs = nn.ModuleList(post_projs)
        # TODO: notice post_projs -> post_projs_vllm, and make sure the state dict is correctly loaded/saved in both vLLM and HuggingFace sides
        self.post_projs_vllm = MergedColumnParallelLinear(
            emb_dim * num_embedders,
            self.hidden_size * num_embedders,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.post_projs_vllm",
        )

    def _shift_right_ignore_eos(
        self, tensor: torch.Tensor, n: int, eos_token_id: int = 2
    ) -> torch.Tensor:
        """Shift tensor right by n positions, resetting at EOS tokens."""
        batch_size, seq_len = tensor.shape
        result = torch.zeros_like(tensor)
        eos_mask = tensor == eos_token_id

        for i in range(batch_size):
            eos_positions = eos_mask[i].nonzero(as_tuple=True)[0]
            prev_idx = 0

            for eos_idx in eos_positions:
                end_idx = eos_idx.item() + 1
                if end_idx - prev_idx > n:
                    result[i, prev_idx + n : end_idx] = tensor[
                        i, prev_idx : end_idx - n
                    ]
                prev_idx = end_idx

            if prev_idx < seq_len and seq_len - prev_idx > n:
                result[i, prev_idx + n : seq_len] = tensor[i, prev_idx : seq_len - n]

        return result

    def _precompute_vocab_mods(self) -> Dict[Tuple[int, int], List[int]]:
        """Precompute modular arithmetic values for vocabulary."""
        if self._vocab_mods_cache is not None:
            return self._vocab_mods_cache

        vocab_mods = {}
        vocab_size = self.vocab_size

        for i in range(2, self.emb_neighbor_num + 1):
            for j in range(self.emb_split_num):
                index = (i - 2) * self.emb_split_num + j
                emb_vocab_dim = int(self.ngram_hash_modulus_base + index * 2 + 1)

                mods = []
                power_mod = 1
                for _ in range(i - 1):
                    power_mod = (power_mod * vocab_size) % emb_vocab_dim
                    mods.append(power_mod)

                vocab_mods[(i, j)] = mods

        self._vocab_mods_cache = vocab_mods
        return vocab_mods

    def _get_ngram_ids(
        self,
        input_ids: torch.Tensor,
        shifted_ids: Dict[int, torch.Tensor],
        vocab_mods: List[int],
        ngram: int,
    ) -> torch.Tensor:
        """Compute N-gram hash IDs using polynomial rolling hash."""
        ngram_ids = input_ids.clone()
        for k in range(2, ngram + 1):
            ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
        return ngram_ids

    def forward(
        self, input_ids: torch.Tensor, ngram_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Stateless forward pass.

        Args:
            input_ids: Current input token IDs of shape (batch_size, seq_len)
            ngram_context: Optional historical context of shape (batch_size, context_len)

        Returns:
            Embedding tensor of shape (batch_size, seq_len, hidden_size)
        """
        seq_len = input_ids.size(-1)

        # Determine complete context
        if ngram_context is not None:
            context = torch.cat(
                [ngram_context[..., -(self.emb_neighbor_num - 1) :], input_ids], dim=-1
            )
        else:
            context = input_ids

        # Base word embeddings
        device = self.word_embeddings.weight.device
        x = self.word_embeddings(input_ids.to(device)).clone()

        # Precompute modular values
        vocab_mods = self._precompute_vocab_mods()

        # Compute shifted IDs
        shifted_ids = {}
        for i in range(2, self.emb_neighbor_num + 1):
            shifted_ids[i] = self._shift_right_ignore_eos(
                context, i - 1, eos_token_id=self.config.eos_token_id
            )

        # Add N-gram embeddings
        for i in range(2, self.emb_neighbor_num + 1):
            for j in range(self.emb_split_num):
                index = (i - 2) * self.emb_split_num + j
                emb_vocab_dim = int(self.ngram_hash_modulus_base + index * 2 + 1)

                ngram_ids = self._get_ngram_ids(
                    context, shifted_ids, vocab_mods[(i, j)], ngram=i
                )
                new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]

                embedder_device = self.embedders[index].weight.device
                x_ngram = self.embedders[index](new_ids.to(embedder_device))

                proj_device = self.post_projs[index].weight.device
                x_proj = self.post_projs[index](x_ngram.to(proj_device))
                x = x + x_proj.to(x.device)

        # Normalize
        x = x / (1 + self.emb_split_num * (self.emb_neighbor_num - 1))

        return x


class LongcatFlashNgramModel(LongcatFlashModel):
    """LongcatFlash model with N-gram enhanced embeddings."""

    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatFlashNgramConfig

    def __init__(self, config):
        super().__init__(config)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.ngram_embeddings = NgramEmbedding(config, self.embed_tokens)

        self.layers = nn.ModuleList(
            [
                LongcatFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_layers)
            ]
        )

        self.head_dim = config.head_dim
        self.config.num_hidden_layers = 2 * config.num_layers
        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LongcatFlashRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # Extract N-gram context if available
        ngram_context = None
        if (
            isinstance(past_key_values, NgramCache)
            and past_key_values.ngram_context is not None
        ):
            ngram_context = past_key_values.ngram_context

        if inputs_embeds is None:
            inputs_embeds = self.ngram_embeddings(
                input_ids, ngram_context=ngram_context
            )

        # Initialize NgramCache if needed
        if use_cache and past_key_values is None:
            past_key_values = NgramCache(config=self.config)

        # Update N-gram context
        if use_cache and isinstance(past_key_values, NgramCache):
            past_key_values.update_ngram_context(input_ids)

        # Prepare cache position
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Forward through decoder layers
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class LongcatFlashNgramForCausalLM(LongcatFlashForCausalLM):
    """LongcatFlash model for causal language modeling with N-gram embeddings."""

    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]
    config_class = LongcatFlashNgramConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LongcatFlashNgramModel(config)

    @torch.no_grad()
    def generate(self, inputs=None, generation_config=None, **kwargs):
        """Override to ensure NgramCache is used."""

        if "past_key_values" not in kwargs or kwargs["past_key_values"] is None:
            kwargs["past_key_values"] = NgramCache(config=self.config)

        return super().generate(
            inputs=inputs, generation_config=generation_config, **kwargs
        )
