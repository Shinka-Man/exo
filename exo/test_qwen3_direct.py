#!/usr/bin/env python3
"""Direct test of Qwen3 MoE model loading."""

import sys
sys.path.insert(0, '/Users/lna/exo/exo')

from pathlib import Path
import mlx.core as mx
from exo.inference.shard import Shard
from exo.inference.mlx.sharded_utils import load_model_shard

def test_qwen3_load():
    model_path = Path("/Users/lna/models/lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-8bit")
    shard = Shard("qwen3-30b-a3b-8bit", 0, 47, 48)
    
    print(f"Loading model from {model_path}")
    print(f"Shard: {shard}")
    
    try:
        model = load_model_shard(model_path, shard, lazy=False)
        print("Model loaded successfully!")
        print(f"Model type: {model.model_type}")
        print(f"Model layers: {len(model.layers)}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    model = test_qwen3_load()
    if model:
        print("\nModel structure:")
        for name, module in model.named_modules():
            print(f"  {name}: {type(module).__name__}")