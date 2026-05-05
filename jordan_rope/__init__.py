"""Complex Jordan-RoPE experiment toolkit.

Heavy PyTorch objects are loaded lazily so config/aggregation utilities can be
used before the training dependencies are installed.
"""

__all__ = [
    "ALiBiBias",
    "CausalTransformerLM",
    "DirectSumRoPEUnipotent",
    "JordanRoPE",
    "RotaryEmbedding",
    "TransformerConfig",
]


def __getattr__(name: str):
    if name in {"CausalTransformerLM", "TransformerConfig"}:
        from .model import CausalTransformerLM, TransformerConfig

        return {"CausalTransformerLM": CausalTransformerLM, "TransformerConfig": TransformerConfig}[name]
    if name in {"ALiBiBias", "DirectSumRoPEUnipotent", "JordanRoPE", "RotaryEmbedding"}:
        from .positional import ALiBiBias, DirectSumRoPEUnipotent, JordanRoPE, RotaryEmbedding

        return {
            "ALiBiBias": ALiBiBias,
            "DirectSumRoPEUnipotent": DirectSumRoPEUnipotent,
            "JordanRoPE": JordanRoPE,
            "RotaryEmbedding": RotaryEmbedding,
        }[name]
    raise AttributeError(name)
