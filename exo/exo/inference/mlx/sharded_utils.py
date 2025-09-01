# Adapted from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py

import glob
import importlib
import json
import logging
import asyncio
import aiohttp
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union, List, Callable
from PIL import Image
from io import BytesIO
import base64
import traceback

import mlx.core as mx
import mlx.nn as nn
import inspect
from transformers import AutoProcessor

from mlx_lm.tokenizer_utils import load_tokenizer, TokenizerWrapper

from exo import DEBUG
from exo.inference.tokenizers import resolve_tokenizer
from ..shard import Shard


class ModelNotFoundError(Exception):
  def __init__(self, message):
    self.message = message
    super().__init__(self.message)


MODEL_REMAPPING = {
  "mistral": "llama",  # mistral is compatible with llama
  "phi-msft": "phixtral",
  # qwen3_moe now has its own implementation
  "kimi_k2": "deepseek_v3",
  "kimi-k2": "deepseek_v3",
}


def _is_kimi_model(config: dict) -> bool:
  mt = str(config.get("model_type", "")).lower()
  return mt in {"kimi_k2", "kimi-k2", "kimi", "moonshot_k2", "moonshot-k2"}


def _infer_kimi_v3_dims_from_weights(config: dict, weights: dict) -> dict:
  """Infer DeepSeek-V3-compatible dims for Kimi-K2 from weight shapes.

  Returns a shallow copy of config with adjusted dimensions. Only fills fields
  it can infer reliably from shapes; leaves the rest unchanged.
  """
  c = dict(config)

  # Locate common prefixes
  prefixes = [
    "model",
    "language_model.model",
  ]
  def first_key(*suffixes):
    for p in prefixes:
      for s in suffixes:
        k = f"{p}.{s}"
        if k in weights:
          return k
    return None

  # Hidden size and vocab size from embeddings or layernorm
  k = first_key("embed_tokens.weight")
  if k is not None:
    vs, hs = weights[k].shape
    c["vocab_size"] = int(vs)
    c["hidden_size"] = int(hs)
  else:
    # Try quantized embeddings to get vocab_size
    ks = first_key("embed_tokens.scales")
    if ks is not None:
      vs, gs = weights[ks].shape
      c["vocab_size"] = int(vs)
    k = first_key("layers.0.input_layernorm.weight")
    if k is not None:
      c["hidden_size"] = int(weights[k].shape[0])

  # Count layers
  max_layer = -1
  for key in weights.keys():
    for p in prefixes:
      pre = f"{p}.layers."
      if key.startswith(pre):
        try:
          idx = int(key.split(".")[2])
          if idx > max_layer:
            max_layer = idx
        except Exception:
          pass
  if max_layer >= 0:
    c["num_hidden_layers"] = max_layer + 1

  # Attention related dims
  # o_proj: (num_heads * v_head_dim, hidden_size)
  okey = first_key("layers.0.self_attn.o_proj.weight")
  qbk = first_key("layers.0.self_attn.q_b_proj.weight")
  qk = first_key("layers.0.self_attn.q_proj.weight")  # when no q_lora_rank
  kvak = first_key("layers.0.self_attn.kv_a_proj_with_mqa.weight")
  kvaln = first_key("layers.0.self_attn.kv_a_layernorm.weight")
  kvbk = first_key("layers.0.self_attn.kv_b_proj.weight")
  qaln = first_key("layers.0.self_attn.q_a_layernorm.weight")

  if okey is not None:
    o_in, o_out = weights[okey].shape
    # hidden_size from o_proj (override to ensure consistency with weights)
    c["hidden_size"] = int(o_out)

    # Decide num_heads
    num_heads = c.get("num_attention_heads")
    if isinstance(num_heads, str):
      try:
        num_heads = int(num_heads)
      except Exception:
        num_heads = None
    if not isinstance(num_heads, int) or num_heads <= 0 or (o_in % num_heads != 0):
      # Heuristic: choose the largest reasonable head count dividing o_in
      for cand in [128, 96, 80, 64, 48, 40, 32, 24, 16, 8]:
        if o_in % cand == 0:
          num_heads = cand
          break
      if not isinstance(num_heads, int) or num_heads <= 0:
        # Fallback to 32
        num_heads = 32
    c["num_attention_heads"] = int(num_heads)
    c["num_key_value_heads"] = int(c.get("num_key_value_heads", num_heads))
    v_head_dim = o_in // num_heads
    c["v_head_dim"] = int(v_head_dim)

  # q path dims
  q_total_out = None
  if qbk is not None:
    q_total_out = int(weights[qbk].shape[1])
    c["q_lora_rank"] = int(weights[qbk].shape[0])
  elif qk is not None:
    q_total_out = int(weights[qk].shape[1])
    c["q_lora_rank"] = None

  if kvaln is not None:
    c["kv_lora_rank"] = int(weights[kvaln].shape[0])
  if kvak is not None and "kv_lora_rank" in c:
    c["qk_rope_head_dim"] = int(weights[kvak].shape[1]) - int(c["kv_lora_rank"])

  if q_total_out is not None and "num_attention_heads" in c:
    q_head_dim = q_total_out // int(c["num_attention_heads"])
    if "qk_rope_head_dim" in c:
      c["qk_nope_head_dim"] = int(q_head_dim) - int(c["qk_rope_head_dim"])
    else:
      # Default split: prefer equal halves if divisible by 2
      if q_head_dim % 2 == 0:
        c["qk_rope_head_dim"] = q_head_dim // 2
        c["qk_nope_head_dim"] = q_head_dim // 2
      else:
        # Put the odd unit to nope
        c["qk_rope_head_dim"] = q_head_dim // 2
        c["qk_nope_head_dim"] = q_head_dim - c["qk_rope_head_dim"]

  # MLP dims
  upk = first_key("layers.0.mlp.up_proj.weight", "layers.0.mlp.switch_mlp.up_proj.weight")
  if upk is not None and "hidden_size" in c:
    sh = weights[upk].shape
    hs = int(c["hidden_size"])
    if sh[0] == hs:
      c["intermediate_size"] = int(sh[1])
    elif sh[1] == hs:
      c["intermediate_size"] = int(sh[0])

  return c


def _get_classes(config: dict):
  """
  Retrieve the model and model args classes based on the configuration.

  Args:
   config (dict): The model configuration.

  Returns:
   A tuple containing the Model class and the ModelArgs class.
  """
  model_type = config["model_type"]
  model_type = MODEL_REMAPPING.get(model_type, model_type)
  try:
    arch = importlib.import_module(f"exo.inference.mlx.models.{model_type}")
  except ImportError:
    msg = f"Model type {model_type} not supported."
    logging.error(msg)
    traceback.print_exc()
    raise ValueError(msg)

  return arch.Model, arch.ModelArgs


def load_config(model_path: Path) -> dict:
  try:
    config_path = model_path / "config.json"
    if config_path.exists():
      with open(config_path, "r") as f:
        config = json.load(f)
      return config
    
    model_index_path = model_path / "model_index.json"
    if model_index_path.exists():
      config = load_model_index(model_path, model_index_path)
      return config
  except FileNotFoundError:
    logging.error(f"Config file not found in {model_path}")
    raise
  return config

def load_model_shard(
  model_path: Path,
  shard: Shard,
  lazy: bool = False,
  model_config: dict = {},
) -> nn.Module:
  """
  Load and initialize the model from a given path.

  Args:
   model_path (Path): The path to load the model from.
   lazy (bool): If False eval the model parameters to make sure they are
    loaded in memory before returning, otherwise they will be loaded
    when needed. Default: ``False``
   model_config(dict, optional): Configuration parameters for the model.
    Defaults to an empty dictionary.

  Returns:
   nn.Module: The loaded and initialized model.

  Raises:
   FileNotFoundError: If the weight files (.safetensors) are not found.
   ValueError: If the model class or args class are not found or cannot be instantiated.
  """
  config = load_config(model_path)
  config.update(model_config)

  # TODO hack
  config["shard"] = {
    "model_id": model_path.name,
    "start_layer": shard.start_layer,
    "end_layer": shard.end_layer,
    "n_layers": shard.n_layers,
  }

  weight_files = glob.glob(str(model_path/"model*.safetensors"))

  if not weight_files:
    # Try weight for back-compat
    weight_files = glob.glob(str(model_path/"weight*.safetensors"))

  # KIMI-K2: infer dims from weights before building the model
  preloaded_weights = None
  if _is_kimi_model(config):
    if not weight_files:
      # Try back-compat filename: also used below as fallback too
      weight_files = glob.glob(str(model_path/"weight*.safetensors"))
    if not weight_files:
      logging.error(f"No safetensors found in {model_path}")
      raise FileNotFoundError(f"No safetensors found in {model_path}")

    # Preload weights to infer shapes once; reuse below to avoid double I/O
    preloaded_weights = {}
    for wf in sorted(weight_files):
      preloaded_weights.update(mx.load(wf))

    # Adjust config in place using inferred dims
    inferred = _infer_kimi_v3_dims_from_weights(config, preloaded_weights)
    if DEBUG >= 6:
      _keys = [
        'hidden_size','num_attention_heads','num_hidden_layers','v_head_dim',
        'qk_rope_head_dim','qk_nope_head_dim','q_lora_rank','kv_lora_rank','intermediate_size'
      ]
      _inferred_subset = {k: inferred[k] for k in _keys if k in inferred}
      print(f"[KIMI-K2] Inferred config overrides: {_inferred_subset}")
    config.update(inferred)

  model_class, model_args_class = _get_classes(config=config)

  class ShardedModel(model_class):
    def __init__(self, args):
      super().__init__(args)
      self.shard = Shard(args.shard.model_id, args.shard.start_layer, args.shard.end_layer, args.shard.n_layers)

    def __call__(self, x, *args, **kwargs):
      y = super().__call__(x, *args, **kwargs)
      return y

  model_args = model_args_class.from_dict(config)
  model = ShardedModel(model_args)

  if config.get("model_index", False):
    model.load()
    return model

  if not weight_files:
    logging.error(f"No safetensors found in {model_path}")
    raise FileNotFoundError(f"No safetensors found in {model_path}")

  weights = preloaded_weights if preloaded_weights is not None else {}
  if preloaded_weights is None:
    for wf in sorted(weight_files):
      if DEBUG >= 8:
        layer_nums = set()
        for k in mx.load(wf):
          if k.startswith("model.layers."):
            layer_num = int(k.split(".")[2])
            layer_nums.add(layer_num)
          if k.startswith("language_model.model.layers."):
            layer_num = int(k.split(".")[3])
            layer_nums.add(layer_num)
        print(f"\"{wf.split('/')[-1]}\": {sorted(layer_nums)},")

      weights.update(mx.load(wf))

  

  if hasattr(model, "sanitize"):
    weights = model.sanitize(weights)
  if DEBUG >= 8:
    print(f"\n|| {config=} ||\n")
  if DEBUG >= 7:
    # Quick sanity: check if quantized keys exist for attention projections
    print(f"config.quantization={config.get('quantization', None)}")
    sample_keys = [
      k for k in weights.keys()
      if k.endswith("self_attn.q_proj.scales") or k.endswith("self_attn.k_proj.scales") or k.endswith("self_attn.v_proj.scales")
    ]
    print(f"Quantization keys present for q/k/v proj: {len(sample_keys) > 0}")

  if (quantization := config.get("quantization", None)) is not None:
    # Sanitize quantization kwargs and support per-module overrides.
    # Some models (e.g., Kimi-K2) store keys like "model.embed_tokens" in the
    # quantization dict. Passing those as kwargs to nn.quantize raises a
    # TypeError. We filter unknown or module-path keys from kwargs and treat
    # them as per-module include/exclude hints handled in class_predicate.

    # Determine allowed kwargs for nn.quantize at runtime (MLX version agnostic).
    allowed_kwargs = {"group_size", "bits"}
    try:
      sig = inspect.signature(nn.quantize)
      # Exclude the first positional argument (model) and our explicit predicate.
      allowed_kwargs = {
        name
        for name, p in sig.parameters.items()
        if name not in {"model", "class_predicate"}
      }
    except Exception:
      pass  # Fall back to default set

    # Normalize common synonyms found in configs
    synonym_map = {
      "q_group_size": "group_size",
      "groupsize": "group_size",
      "group": "group_size",
      "q_bits": "bits",
      "n_bits": "bits",
      "nbits": "bits",
      "wbits": "bits",
    }

    def normalize_key(k: str) -> str:
      return synonym_map.get(k, k)

    # Split into top-level kwargs vs per-module overrides (module path contains '.')
    q_kwargs = {}
    module_overrides = {}
    if isinstance(quantization, dict):
      for k, v in quantization.items():
        if not isinstance(k, str):
          continue
        nk = normalize_key(k)
        if "." in nk or nk.startswith("model.") or nk.startswith("language_model."):
          module_overrides[nk] = v
          continue
        if nk in allowed_kwargs:
          q_kwargs[nk] = v
    else:
      # If quantization is not a dict, do nothing special and let nn.quantize decide
      q_kwargs = {}
      module_overrides = {}

    # Interpret override values per-module: can return False (skip), True (use defaults), or a params dict
    def _canon(s: str) -> str:
      # Normalize common prefixes to improve matching between leaf paths and weight keys
      for pref in ("language_model.model.", "language_model.", "model."):
        if s.startswith(pref):
          return s[len(pref):]
      return s

    def override_decider(p: str):
      p_can = _canon(p)
      for key, val in module_overrides.items():
        k_can = _canon(key)
        if p == key or p.startswith(key + ".") or p_can == k_can or p_can.startswith(k_can + "."):
          # Bool => skip/include explicitly
          if isinstance(val, bool):
            return val
          # None => skip
          if val is None:
            return False
          # Numeric => treat as bits unless it's a float16/32 sentinel
          if isinstance(val, (int, float)):
            if val in (0, 16, 32):
              return False
            return {"bits": int(val)}
          # String
          if isinstance(val, str):
            s = val.lower().strip()
            if s in {"skip", "exclude", "none", "fp16", "fp32", "no", "off"}:
              return False
            if s.isdigit():
              return {"bits": int(s)}
            return True
          # Dict => sanitize keys as params
          if isinstance(val, dict):
            if any(val.get(k, False) for k in ("skip", "disable", "exclude")):
              return False
            params = {}
            for kk, vv in val.items():
              nk = normalize_key(str(kk))
              if nk in {"bits", "group_size", "mode"}:
                params[nk] = vv
            # If contains any param, return the dict, else include by default
            return params if params else True
          # Unknown type => include
          return True
      return None

    # Precompute per-module params from weights for robust matching (handles lm_head, embed, q/k/v, mlp, etc.)
    def _canon_path_for_key(k: str) -> str:
      # Strip well-known prefixes and trailing field suffix
      base = k
      for suf in (".scales", ".weight", ".biases"):
        if base.endswith(suf):
          base = base[: -len(suf)]
      for pref in ("language_model.model.", "language_model.", "model."):
        if base.startswith(pref):
          base = base[len(pref):]
      return base

    per_module_params = {}
    try:
      # Build a quick lookup set of keys for existence testing
      keyset = set(weights.keys())
      # Scan all scales keys and derive params when matching weight exists
      for k in keyset:
        if not isinstance(k, str) or not k.endswith('.scales'):
          continue
        base = k[:-7]
        wkey = f"{base}.weight"
        if wkey not in keyset:
          continue
        # Derive params
        try:
          G = int(weights[k].shape[1])
          P = int(weights[wkey].shape[1])
        except Exception:
          continue
        if G <= 0 or P <= 0:
          continue
        # Candidate sets, prefer global hints
        gs_candidates = [64, 128, 32, 16, 8]
        bit_candidates = [4, 3, 2, 8, 6]
        gsg = q_kwargs.get("group_size")
        if isinstance(gsg, int) and gsg in gs_candidates:
          gs_candidates = [gsg] + [g for g in gs_candidates if g != gsg]
        bbg = q_kwargs.get("bits")
        if isinstance(bbg, int) and bbg in bit_candidates:
          bit_candidates = [bbg] + [b for b in bit_candidates if b != bbg]
        found = None
        for gs in gs_candidates:
          for b in bit_candidates:
            if P * 32 == G * gs * b:
              found = {"group_size": gs, "bits": b}
              if "mode" in q_kwargs:
                found["mode"] = q_kwargs["mode"]
              break
          if found:
            break
        if found:
          per_module_params[_canon_path_for_key(base)] = found
    except Exception as _e:
      if DEBUG >= 2:
        print(f"[quantize] per-module params precompute failed: {_e}")

    # Try to derive per-module quantization params from weight shapes lazily
    def derive_params_from_weights(path: str):
      # Build candidate key prefixes to bridge differences between leaf paths and stored weight keys
      cands = []
      cands.append(path)
      if not path.startswith("model."):
        cands.append(f"model.{path}")
      if path.startswith("layers."):
        cands.append(f"model.{path}")
      # Some repos store under language_model.model.*
      cands.append(f"language_model.model.{path}")

      found = None
      for base in cands:
        scales_k = f"{base}.scales"
        weight_k = f"{base}.weight"
        if scales_k in weights and weight_k in weights:
          found = (scales_k, weight_k)
          break
      if not found:
        # Try the precomputed table using canonicalized path
        canon = _canon(path)
        return per_module_params.get(canon, None)
      scales_k, weight_k = found
      try:
        G = int(weights[scales_k].shape[1])
        P = int(weights[weight_k].shape[1])
      except Exception:
        return None
      if G <= 0 or P <= 0:
        return None
      # Candidate sets
      gs_candidates = [64, 128, 32, 16, 8]
      bit_candidates = [6, 4, 3, 2, 8]
      # If global hints exist, prefer them by ordering
      gsg = q_kwargs.get("group_size")
      if isinstance(gsg, int) and gsg in gs_candidates:
        gs_candidates = [gsg] + [g for g in gs_candidates if g != gsg]
      bbg = q_kwargs.get("bits")
      if isinstance(bbg, int) and bbg in bit_candidates:
        bit_candidates = [bbg] + [b for b in bit_candidates if b != bbg]
      for gs in gs_candidates:
        for b in bit_candidates:
          if P * 32 == G * gs * b:
            params = {"group_size": gs, "bits": b}
            if "mode" in q_kwargs:
              params["mode"] = q_kwargs["mode"]
            if DEBUG >= 5:
              print(f"[quantize] derive_params_from_weights: path={path} -> {params} (P={P}, G={G})")
            return params
      # Fallback to precomputed mapping if direct solve failed
      result = per_module_params.get(_canon(path), None)
      if DEBUG >= 5:
        print(f"[quantize] derive_params_from_weights fallback: path={path} -> {result} (P={P}, G={G})")
      return result

    # Auto-infer a compatible group_size from weights if needed
    def infer_group_size_from_scales(hidden_size: int) -> Optional[int]:
      candidates = {}
      for k, v in weights.items():
        if not isinstance(k, str) or not k.endswith(".scales"):
          continue
        shp = getattr(v, "shape", None)
        if not shp or len(shp) != 2:
          continue
        for dim in shp:
          if isinstance(dim, int) and dim > 0 and hidden_size % int(dim) == 0:
            g = hidden_size // int(dim)
            # Filter to common group sizes
            if g in {8, 16, 32, 64, 128}:
              candidates[g] = candidates.get(g, 0) + 1
      if candidates:
        # Prefer common sizes first, then larger ones for better perf
        return sorted(candidates.items(), key=lambda x: (-x[1], -x[0]))[0][0]
      return None

    # If hidden_size is available, make group_size consistent with weights
    auto_group = None
    try:
      hs = int(config.get("hidden_size", 0))
      if hs:
        auto_group = infer_group_size_from_scales(hs)
        if auto_group:
          gs = q_kwargs.get("group_size")
          if gs is None or (isinstance(gs, int) and (hs % gs != 0 or gs != auto_group)):
            if DEBUG >= 6:
              print(f"[quantize] Adjusting group_size -> {auto_group} (hidden_size={hs})")
            q_kwargs["group_size"] = int(auto_group)
    except Exception as _e:
      if DEBUG >= 2:
        print(f"[quantize] group_size auto-infer failed: {_e}")

    # Infer bits from embed_tokens (or a compatible module) if possible
    def get_key(*suffixes):
      prefixes = ("model", "language_model.model")
      for pre in prefixes:
        for suf in suffixes:
          k = f"{pre}.{suf}"
          if k in weights:
            return k
      return None

    try:
      emb_scales_k = get_key("embed_tokens.scales")
      emb_weight_k = get_key("embed_tokens.weight")
      if emb_scales_k and emb_weight_k and hs:
        G = int(weights[emb_scales_k].shape[1])
        # Prefer the chosen/auto group size if available
        gs = int(q_kwargs.get("group_size") or (hs // G))
        packed_per_row = int(weights[emb_weight_k].shape[1])
        # ints per group = packed_per_row / G; bits = (ints_per_group * 32) / group_size
        ints_per_group = packed_per_row / G
        inferred_bits = int(round((ints_per_group * 32) / gs))
        if inferred_bits > 0:
          cur_bits = q_kwargs.get("bits")
          if (not isinstance(cur_bits, int)) or cur_bits != inferred_bits:
            if DEBUG >= 6:
              print(f"[quantize] Adjusting bits -> {inferred_bits} (derived from embed_tokens)")
            q_kwargs["bits"] = inferred_bits
    except Exception as _e:
      if DEBUG >= 2:
        print(f"[quantize] bits auto-infer failed: {_e}")

    # Handle legacy models which may not have everything quantized or have per-layer bitwidths
    def class_predicate(p, m):
      if not hasattr(m, "to_quantized"):
        return False
      # Respect explicit per-module overrides if provided
      decision = override_decider(p) if module_overrides else None
      if isinstance(decision, dict):
        return decision
      if isinstance(decision, bool):
        return decision
      # Derive params from stored quantized weights
      derived = derive_params_from_weights(p)
      if derived is not None:
        return derived
      # Auto-detect: quantize only modules with corresponding quantized weights
      return f"{p}.scales" in weights

    # Finally call quantize with sanitized kwargs
    nn.quantize(
      model,
      **q_kwargs,
      class_predicate=class_predicate,
    )
    if DEBUG >= 7:
      # Verify a known layer got quantized by checking param keys
      first_layer = 0
      try:
        proj = model.layers[first_layer].self_attn.q_proj
        print(f"q_proj params after quantize: {list(proj.parameters().keys())}")
      except Exception as e:
        print(f"Failed to introspect q_proj params: {e}")

  model.load_weights(list(weights.items()), strict=True)

  if not lazy:
    mx.eval(model.parameters())

  model.eval()
  return model

async def load_shard(
  model_path: str,
  shard: Shard,
  tokenizer_config={},
  model_config={},
  adapter_path: Optional[str] = None,
  lazy: bool = False,
) -> Tuple[nn.Module, TokenizerWrapper]:
  model = load_model_shard(model_path, shard, lazy, model_config)

  # TODO: figure out a generic solution
  if model.model_type == "llava":
    processor = AutoProcessor.from_pretrained(model_path)
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.encode = processor.tokenizer.encode
    return model, processor
  elif hasattr(model, "tokenizer"):
    tokenizer = model.tokenizer
    return model, tokenizer
  else:
    tokenizer = await resolve_tokenizer(model_path)
    return model, tokenizer


async def get_image_from_str(_image_str: str):
  image_str = _image_str.strip()

  if image_str.startswith("http"):
    async with aiohttp.ClientSession() as session:
      async with session.get(image_str, timeout=10) as response:
        content = await response.read()
        return Image.open(BytesIO(content)).convert("RGB")
  elif image_str.startswith("data:image/"):
    # Extract the image format and base64 data
    format_prefix, base64_data = image_str.split(";base64,")
    image_format = format_prefix.split("/")[1].lower()
    if DEBUG >= 2: print(f"{image_str=} {image_format=}")
    imgdata = base64.b64decode(base64_data)
    img = Image.open(BytesIO(imgdata))

    # Convert to RGB if not already
    if img.mode != "RGB":
      img = img.convert("RGB")

    return img
  else:
    raise ValueError("Invalid image_str format. Must be a URL or a base64 encoded image.")

# loading a combined config for all models in the index
def load_model_index(model_path: Path, model_index_path: Path):
  models_config = {}
  with open(model_index_path, "r") as f:
      model_index = json.load(f)
  models_config["model_index"] = True
  models_config["model_type"] = model_index["_class_name"]
  models_config["models"] = {}
  for model in model_index.keys():
    model_config_path = glob.glob(str(model_path / model / "*config.json"))
    if len(model_config_path)>0:
      with open(model_config_path[0], "r") as f:
        model_config = { }
        model_config["model_type"] = model
        model_config["config"] = json.load(f)
        model_config["path"] = model_path / model
        if model_config["path"]/"*model.safetensors":
          model_config["config"].update({"weight_files": list(glob.glob(str(model_config["path"]/"*model.safetensors")))})
        model_config["path"] = str(model_path / model)
        m = {}
        m[model] = model_config
        models_config.update(m)
  return models_config
