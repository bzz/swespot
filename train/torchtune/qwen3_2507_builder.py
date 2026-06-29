"""Custom torchtune model builder for **Qwen3-4B-Instruct-2507**.

torchtune's stock ``lora_qwen3_4b_instruct`` targets the original ``Qwen/Qwen3-4B``
and hardcodes ``rope_base=1000000.0`` / ``max_seq_len=40960``. The *2507* instruct
checkpoint is the long-context variant: its ``config.json`` reports
``rope_theta=5000000`` and ``max_position_embeddings=262144``. Using the stock
builder would silently mismatch the RoPE frequencies against the trained weights.

This builder is a faithful copy of ``lora_qwen3_4b_instruct`` with ``rope_base``
set to ``5e6`` (and ``max_seq_len`` capped to our training length to keep the RoPE
cache small — RoPE *base*, not *max_seq_len*, is what must match the weights).

Referenced from the recipe config as
``_component_: qwen3_2507_builder.lora_qwen3_4b_instruct_2507`` with
``PYTHONPATH`` including this directory.
"""

from torchtune.models.qwen3._component_builders import lora_qwen3

# All dims verified against the downloaded Qwen3-4B-Instruct-2507 config.json:
# vocab 151936, layers 36, heads 32, kv 8, embed 2560, intermediate 9728,
# head_dim 128, norm_eps 1e-6, tie_word_embeddings True, rope_theta 5e6.
ROPE_BASE_2507 = 5_000_000.0
TRAIN_MAX_SEQ_LEN = 32768  # our max_length; <= the model's 262144 context


def lora_qwen3_4b_instruct_2507(
    lora_attn_modules,
    apply_lora_to_mlp: bool = False,
    apply_lora_to_output: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_dropout: float = 0.0,
    use_dora: bool = False,
    quantize_base: bool = False,
    max_seq_len: int = TRAIN_MAX_SEQ_LEN,
):
    """Qwen3-4B-Instruct-2507 with LoRA — identical to torchtune's 4B-instruct
    builder except ``rope_base=5e6`` (the 2507 value) and a configurable
    ``max_seq_len`` (default 32768)."""
    return lora_qwen3(
        lora_attn_modules=lora_attn_modules,
        apply_lora_to_mlp=apply_lora_to_mlp,
        apply_lora_to_output=apply_lora_to_output,
        vocab_size=151936,
        num_layers=36,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2560,
        intermediate_dim=9728,
        max_seq_len=max_seq_len,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=ROPE_BASE_2507,
        q_proj_bias=False,
        k_proj_bias=False,
        v_proj_bias=False,
        q_norm=True,
        k_norm=True,
        tie_word_embeddings=True,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_dora=use_dora,
        quantize_base=quantize_base,
    )
