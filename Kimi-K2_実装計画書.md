# Kimi-K2 1兆パラメータモデル MLX実装計画書

## 1. エラー分析

### 現在の問題
```
Expected shape (896) but received (672)
```

### 原因の特定
- **期待される次元**: 896 
- **実際の次元**: 672
- **差分**: 224次元

### モデル仕様から判明したこと
```json
{
  "hidden_size": 7168,
  "num_attention_heads": 64,
  "q_lora_rank": 1536,  // Query LoRA rank
  "kv_lora_rank": 512,  // Key-Value LoRA rank
  "qk_nope_head_dim": 128,  // Non-positional head dimension
  "qk_rope_head_dim": 64    // Rotary positional head dimension
}
```

### 次元計算の分析
- **標準的なDeepSeek-V3**: `head_dim = hidden_size / num_heads = 7168 / 64 = 112`
- **Kimi-K2の特殊構造**: 
  - nope_head_dim (128) + rope_head_dim (64) = 192次元/ヘッド
  - しかし実際は LoRA による圧縮が適用されている

## 2. 実装戦略

### Option A: deepseek_v3.pyの修正 (短期的解決)
**メリット**: 
- 既存のコードベースを活用
- 最小限の変更で動作可能

**デメリット**:
- DeepSeek-V3の本来の動作に影響する可能性
- Kimi-K2特有の最適化が活かせない

### Option B: kimi_k2.py専用実装 (推奨)
**メリット**:
- Kimi-K2の独自アーキテクチャを完全サポート
- 1兆パラメータに最適化された実装
- 将来的な改善が容易

**デメリット**:
- 実装工数が大きい
- テストとデバッグが必要

## 3. 実装計画

### Phase 1: 調査とプロトタイプ (即座に実施)
1. **モデル重みの調査**
   - 実際のweight tensorの形状を確認
   - LoRA分解された重みの構造を理解
   
2. **DeepSeek-V3実装の分析**
   - attention層の実装詳細を確認
   - 672次元と896次元の不一致箇所を特定

### Phase 2: Kimi-K2モデル実装 (1-2時間)
1. **基本構造の実装**
   ```python
   # exo/inference/mlx/models/kimi_k2.py
   class KimiK2Attention(nn.Module):
       def __init__(self, args):
           super().__init__()
           self.q_lora_rank = args.q_lora_rank  # 1536
           self.kv_lora_rank = args.kv_lora_rank  # 512
           # LoRA decomposed projections
           self.q_a_proj = nn.Linear(args.hidden_size, self.q_lora_rank, bias=False)
           self.q_b_proj = nn.Linear(self.q_lora_rank, args.hidden_size, bias=False)
           # ...
   ```

2. **LoRA圧縮の実装**
   - Query: 7168 → 1536 → 7168
   - Key/Value: 7168 → 512 → 7168

3. **MoE層の実装**
   - 384 experts (n_routed_experts)
   - Top-8 routing (num_experts_per_tok)

### Phase 3: 統合とテスト (30分)
1. **models.pyへの登録**
2. **小規模テスト** (1-2層のみロード)
3. **メモリプロファイリング**

### Phase 4: 分散推論の最適化 (30分)
1. **層の分割戦略**
   - M2 (192GB): 0-16層
   - M3 (512GB): 17-60層
2. **通信最適化**
3. **推論速度の測定**

## 4. 実装の詳細設計

### 4.1 Attention機構の修正
```python
class KimiK2Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        
        # LoRA decomposed Query projection
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.hidden_size, bias=False)
        
        # LoRA decomposed KV projection with MQA
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, 
            self.kv_lora_rank + self.qk_rope_head_dim, 
            bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, 
            2 * self.hidden_size, 
            bias=False
        )
        
        # Output projection
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rope = nn.RoPE(
            self.qk_rope_head_dim,
            traditional=False,
            base=10000
        )
```

### 4.2 MoE層の実装
```python
class KimiK2MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_routed_experts = args.n_routed_experts  # 384
        self.n_shared_experts = args.n_shared_experts  # 1
        self.num_experts_per_tok = args.num_experts_per_tok  # 8
        
        # Router
        self.gate = nn.Linear(args.hidden_size, self.n_routed_experts, bias=False)
        
        # Shared expert
        self.shared_expert = KimiK2MLP(args, is_shared=True)
        
        # Routed experts
        self.experts = nn.ModuleList([
            KimiK2MLP(args, expert_id=i) 
            for i in range(self.n_routed_experts)
        ])
```

## 5. テスト計画

### 5.1 単体テスト
1. **Attention層テスト**
   - 入力形状: [batch, seq_len, 7168]
   - 出力形状: [batch, seq_len, 7168]

2. **MoE層テスト**
   - Top-K routing検証
   - Expert選択の確認

### 5.2 統合テスト
1. **1層のみでの動作確認**
2. **メモリ使用量の検証**
3. **推論速度の測定**

### 5.3 分散テスト
1. **M2側で0-5層のみロード**
2. **層間の通信確認**
3. **完全な推論パイプライン**

## 6. リスクと対策

### リスク1: メモリ不足
**対策**: 
- 層ごとの段階的ロード
- より積極的な量子化（3bit）

### リスク2: 推論速度の低下
**対策**:
- Expert並列化の実装
- キャッシュ最適化

### リスク3: 精度の劣化
**対策**:
- 量子化パラメータの調整
- 重要な層のみFP16維持

## 7. 実装スケジュール

| フェーズ | 作業内容 | 所要時間 | 完了基準 |
|---------|---------|----------|----------|
| 1 | Weight形状調査 | 15分 | 実際の次元を特定 |
| 2 | kimi_k2.py作成 | 60分 | コンパイル成功 |
| 3 | Attention実装 | 30分 | 単体テスト通過 |
| 4 | MoE実装 | 30分 | 単体テスト通過 |
| 5 | 統合テスト | 30分 | 1層での推論成功 |
| 6 | 分散推論 | 30分 | 2ノード間で動作 |

## 8. 成功基準

1. **技術的成功**
   - 形状エラーの解消
   - メモリ使用量 < 160GB (M2側)
   - 推論可能な状態

2. **性能目標**
   - トークン/秒 > 1
   - 初回ロード時間 < 5分

3. **品質基準**
   - 生成テキストの一貫性
   - 40Kトークンコンテキストの処理

## 9. Codexとの協業ポイント

### Codexが担当する部分
1. **詳細な実装**
   - kimi_k2.pyの完全実装
   - 複雑な数学的変換

2. **最適化**
   - SIMD命令の活用
   - メモリアクセスパターン

### 当方が担当する部分
1. **テストとデバッグ**
   - 実機での動作確認
   - エラーログの収集

2. **統合作業**
   - exoフレームワークへの組み込み
   - M2/M3間の調整

## 10. 次のステップ

1. **即座に実施**
   - Weight tensorの形状調査
   - 672次元の出所を特定

2. **Codexと共有**
   - この計画書をレビュー
   - 実装方針の合意

3. **実装開始**
   - kimi_k2.pyのスケルトン作成
   - 段階的な肉付け

---

*"1兆パラメータの壁を越える、歴史的な瞬間へ"*