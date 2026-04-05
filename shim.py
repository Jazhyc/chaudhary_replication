"""
HookedTransformerShim — architecture-agnostic wrapper around any HuggingFace causal LM.

Mimics the TransformerLens HookedTransformer API so existing pipeline code
(vector_generation.py, probe_analysis.py) works unchanged with new model families.

Extracted from original_code/probe_eval.ipynb (Chaudhary et al. 2025).
"""

import torch
import re
from types import SimpleNamespace
from contextlib import contextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from typing import Optional, List, Tuple, Union, Dict, Any


class HookedTransformerShim:
    """
    Minimal API-compatible shim for the HookedTransformer used in transformer_lens.
    Works for extracting activations with forward hooks (read-only).
    Uses HuggingFace's output_hidden_states=True to capture residual stream activations.
    """

    def __init__(self, hf_model, tokenizer, device, cfg):
        self.model = hf_model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.cfg = cfg
        self._hooks = {}
        self._hook_handles = []

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu",
                        dtype: Optional[torch.dtype] = None,
                        revision: Optional[str] = None):
        """Load a pretrained model from Hugging Face hub."""
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if getattr(tok, "pad_token", None) is None:
            if getattr(tok, "eos_token", None) is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.pad_token = tok.unk_token

        # Load config with hidden-state output enabled
        cfg_hf = AutoConfig.from_pretrained(model_path)
        cfg_hf.output_hidden_states = True
        cfg_hf.output_attentions = True
        cfg_hf.return_dict = True

        # Prepare model loading kwargs
        load_kwargs = {"config": cfg_hf}
        if revision is not None:
            load_kwargs["revision"] = revision

        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype

        # For CUDA multi-GPU, use device_map="auto" (requires accelerate).
        # For CPU/MPS/single-GPU, just load and .to(device).
        if device == "cuda" and torch.cuda.device_count() > 1:
            load_kwargs["device_map"] = "auto"

        print(f"Downloading/loading model weights...")
        model_hf = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        if "device_map" not in load_kwargs:
            model_hf.to(device)

        # --- Infer architecture properties ---
        # For models with nested text config (e.g. Gemma 3)
        text_cfg = getattr(cfg_hf, "text_config", cfg_hf)

        n_layers = (
            getattr(text_cfg, "num_hidden_layers", None)
            or getattr(text_cfg, "n_layers", None)
            or getattr(text_cfg, "num_layers", None)
            or getattr(text_cfg, "n_layer", None)
        )
        if n_layers is None and hasattr(model_hf, "model"):
            if hasattr(model_hf.model, "layers"):
                n_layers = len(model_hf.model.layers)
            elif hasattr(model_hf.model, "h"):
                n_layers = len(model_hf.model.h)

        d_model = (
            getattr(text_cfg, "hidden_size", None)
            or getattr(text_cfg, "d_model", None)
            or getattr(text_cfg, "n_embd", None)
            or getattr(text_cfg, "dim", None)
        )

        n_heads = (
            getattr(text_cfg, "num_attention_heads", None)
            or getattr(text_cfg, "n_heads", None)
            or getattr(text_cfg, "num_heads", None)
            or getattr(text_cfg, "n_head", None)
        )

        # Fallback: infer d_model from the embedding layer weights
        if d_model is None and hasattr(model_hf, "model"):
            if hasattr(model_hf.model, "embed_tokens"):
                d_model = model_hf.model.embed_tokens.weight.shape[1]
            elif hasattr(model_hf.model, "wte"):
                d_model = model_hf.model.wte.weight.shape[1]

        # Fallback: infer n_heads from first layer's attention module
        if n_heads is None and hasattr(model_hf, "model"):
            layers_attr = getattr(model_hf.model, "layers", None) or getattr(model_hf.model, "h", None)
            if layers_attr and len(layers_attr) > 0:
                first_layer = layers_attr[0]
                attn = getattr(first_layer, "self_attn", None) or getattr(first_layer, "attn", None)
                if attn is not None:
                    n_heads = getattr(attn, "num_heads", None) or getattr(attn, "n_heads", None)

        if n_layers is None:
            raise ValueError(
                f"Could not determine number of layers for model {model_path}."
            )

        cfg = SimpleNamespace(
            model_name=model_path,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_model // n_heads if n_heads else None,
            d_vocab=getattr(text_cfg, "vocab_size", None),
            n_ctx=(
                getattr(text_cfg, "max_position_embeddings", None)
                or getattr(text_cfg, "n_positions", None)
            ),
            eps=getattr(text_cfg, "layer_norm_epsilon",
                        getattr(text_cfg, "rms_norm_eps", 1e-5)),
            use_attn_result=True,
            use_hook_tokens=True,
        )

        print(f"Loaded {model_path}: {n_layers} layers, d_model={d_model}, "
              f"n_heads={n_heads}")

        return cls(model_hf, tok, device, cfg)

    def model_eval(self):
        """Set model to inference mode."""
        self.model.eval()
        return self

    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def reset_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._hooks = {}

    @contextmanager
    def hooks(self, fwd_hooks: Optional[List[Tuple[str, callable]]] = None,
              **kwargs):
        """Context manager that registers hooks for the next forward pass."""
        if fwd_hooks is None:
            fwd_hooks = []

        # Store hooks to be applied during __call__
        for name, fn in fwd_hooks:
            self._hooks[name] = fn

        try:
            yield {}
        finally:
            self._hooks = {}

    def to_tokens(self, prompt: Union[str, List[str]],
                  prepend_bos: bool = True) -> torch.Tensor:
        if isinstance(prompt, (list, tuple)):
            ids = []
            for p in prompt:
                encoded = self.tokenizer.encode(p, add_special_tokens=prepend_bos)
                ids.append(encoded)
            max_len = max(len(seq) for seq in ids)
            padded = []
            for seq in ids:
                if len(seq) < max_len:
                    seq = seq + [self.tokenizer.pad_token_id] * (max_len - len(seq))
                padded.append(seq)
            return torch.tensor(padded).to(self.device)
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=prepend_bos)
            return torch.tensor([ids]).to(self.device)

    def to_string(self, tokens: torch.Tensor) -> Union[str, List[str]]:
        if tokens.dim() == 1:
            return self.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
        else:
            return [self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                    for seq in tokens]

    def to_single_token(self, string: str) -> int:
        return self.tokenizer.encode(string, add_special_tokens=False)[0]

    def __call__(self, tokens: torch.Tensor, **kwargs) -> Any:
        """Forward pass. Applies registered hooks via output_hidden_states."""
        if isinstance(tokens, dict):
            inputs = {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in tokens.items()
            }
        else:
            inputs = {"input_ids": tokens.to(self.device)}

        inputs.update(kwargs)

        # Always request hidden states when hooks are registered
        if self._hooks:
            inputs["output_hidden_states"] = True

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Dispatch hooks using hidden_states
        if self._hooks and hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states

            for name, fn in self._hooks.items():
                activation = None

                match_pre = re.match(r"blocks\.(\d+)\.hook_resid_pre", name)
                if match_pre:
                    layer_idx = int(match_pre.group(1))
                    if layer_idx < len(hidden_states):
                        activation = hidden_states[layer_idx]

                match_post = re.match(r"blocks\.(\d+)\.hook_resid_post", name)
                if match_post:
                    layer_idx = int(match_post.group(1))
                    if layer_idx + 1 < len(hidden_states):
                        activation = hidden_states[layer_idx + 1]

                if activation is not None:
                    hook_point = SimpleNamespace(name=name)
                    try:
                        fn(activation, hook_point)
                    except TypeError:
                        fn(activation)

        return outputs
