"""Model loading.

load_model_hf loads a HuggingFace AutoModelForCausalLM; it is used by every
stage (greedy decoding, attribution patching, activation patching). It returns (model, tokenizer, config), where
config is a dict with the keys required by attribution.py (num_layers, num_heads, head_dim, intermediate_size,
family, layers_path, mlp_hook, attn_hook).
"""

import torch


def _detect_family(model_name):
    """All currently supported models share the Llama-style decoder layout
    (model.model.layers, mlp.down_proj, self_attn.o_proj). The 'family' string
    is informational; the hook paths in get_mlp_hook / get_attn_hook are
    identical for all of these architectures.

    Mixture-of-experts models (e.g. Qwen3-30B-A3B-Base) are NOT supported by
    this pipeline — `mlp.down_proj` doesn't exist on MoE blocks; they have
    expert.down_proj per expert. Caller should skip them.
    """
    name = model_name.lower()
    if "moe" in name or "a3b" in name:
        raise ValueError(
            f"MoE model detected ({model_name}). The MLP attribution hook "
            f"targets a single down_proj per layer; MoE has one per expert. "
            f"Skip this model or extend src/attribution.py to handle experts."
        )
    for fam in ("qwen", "mistral", "olmo", "gemma", "phi", "llama"):
        if fam in name:
            return fam
    raise ValueError(
        f"Unsupported model: {model_name}. Add a branch in src/model.py."
    )


def _build_config(hf_config, model_name):
    family = _detect_family(model_name)
    num_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)
    return {
        "model_name": model_name,
        "family": family,
        "num_layers": hf_config.num_hidden_layers,
        "num_heads": num_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "intermediate_size": hf_config.intermediate_size,
        "layers_path": "model.layers",
        "mlp_hook": "mlp.down_proj.input",
        "attn_hook": "self_attn.o_proj.input",
        "lm_head_path": "lm_head",
    }


def load_model_hf(model_name, dtype=torch.float16, device_map="auto"):
    """HuggingFace loader used for stage 1 (greedy decode) and stage 2
    (attribution patching with raw forward-pre-hooks).

    Params are frozen to avoid storing param-grads during stage 2's backward —
    we only need grads on the hook activations (which we enable in-hook).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF model: {model_name} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    config = _build_config(model.config, model_name)
    print(f"  loaded: {config['num_layers']} layers, "
          f"{config['num_heads']} heads, "
          f"intermediate={config['intermediate_size']}")
    return model, tokenizer, config


def get_layers(model, model_config):
    return model.model.layers


def get_mlp_hook(layer, model_config):
    return layer.mlp.down_proj.input


def get_attn_hook(layer, model_config):
    return layer.self_attn.o_proj.input
