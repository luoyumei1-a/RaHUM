"""Model-level exports with lazy loading for optional heavy dependencies."""

__all__ = [
    "CHRLMMLlamaForCausalLM",
    "CHRLMMConfig",
    "PointTransformer",
    "CHRLMMHallucinationReducer",
]


def __getattr__(name):
    if name in {"CHRLMMLlamaForCausalLM", "CHRLMMConfig"}:
        from .CHRLMM import CHRLMMLlamaForCausalLM, CHRLMMConfig

        return {
            "CHRLMMLlamaForCausalLM": CHRLMMLlamaForCausalLM,
            "CHRLMMConfig": CHRLMMConfig,
        }[name]
    if name == "PointTransformer":
        from .pointbert.point_encoder import PointTransformer

        return PointTransformer
    if name == "CHRLMMHallucinationReducer":
        from .CHRLMM_hallucination import CHRLMMHallucinationReducer

        return CHRLMMHallucinationReducer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
