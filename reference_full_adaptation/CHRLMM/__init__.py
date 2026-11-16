"""CHR-LMM package exports.

Heavy language-model dependencies are imported lazily so the dataset tools,
synthetic benchmark, and hallucination reducer can run in lightweight
environments without requiring the full LMM stack.
"""

__all__ = ["CHRLMMLlamaForCausalLM", "CHRLMMConfig", "CHRLMMHallucinationReducer"]


def __getattr__(name):
    if name in {"CHRLMMLlamaForCausalLM", "CHRLMMConfig"}:
        from .model.CHRLMM import CHRLMMLlamaForCausalLM, CHRLMMConfig

        return {
            "CHRLMMLlamaForCausalLM": CHRLMMLlamaForCausalLM,
            "CHRLMMConfig": CHRLMMConfig,
        }[name]
    if name == "CHRLMMHallucinationReducer":
        from .model.CHRLMM_hallucination import CHRLMMHallucinationReducer

        return CHRLMMHallucinationReducer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
