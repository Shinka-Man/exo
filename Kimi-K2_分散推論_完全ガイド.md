# Kimi-K2 1兆パラメータモデル分散推論 完全ガイド

## 🚀 概要
本ガイドは、Moonshot AI製のKimi-K2（1兆パラメータ、504GB）モデルを、Apple Silicon Mac 2台で分散推論する方法を詳細に解説します。

### 達成内容
- **モデル**: Kimi-K2-Instruct-MLX-3.985bit（504GB）
- **ハードウェア**: M2 Studio (192GB) + M3 Ultra (512GB)
- **フレームワーク**: exo + MLX
- **実績**: 完全動作、トークン生成成功

## 📋 前提条件

### ハードウェア要件
- **Mac 1**: 192GB以上のメモリ（M2 Ultra推奨）
- **Mac 2**: 512GB以上のメモリ（M3 Ultra推奨）
- **接続**: Thunderbolt 4ケーブル（10Gbps以上）

### ソフトウェア要件
- macOS 14.0以上
- Python 3.10以上
- 500GB以上の空きディスク容量

## 🔧 セットアップ

### 1. exoフレームワークのインストール

両方のMacで以下を実行：

```bash
# リポジトリのクローン
git clone https://github.com/exo-explore/exo.git
cd exo

# Python仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -e .

# 追加の依存関係（Kimi-K2用）
pip install tiktoken blobfile
```

### 2. モデルの配置

両方のMacで同じパスにモデルを配置：

```bash
# モデルディレクトリの作成
mkdir -p ~/models/inferencerlabs

# モデルをダウンロード（HuggingFaceから）
# または既存のモデルをコピー
cp -r /path/to/Kimi-K2-Instruct-MLX-3.985bit ~/models/inferencerlabs/
```

### 3. 必須パッチの適用

#### 3.1 sharded_utils.py の修正

`exo/exo/inference/mlx/sharded_utils.py` の460行目付近に以下のコードを追加：

```python
# 460行目付近、quantize関数内に追加
# Precompute per-module params from weights
per_module_params = {}
try:
    def _canon(path):
        # Canonicalize path
        if path.startswith("model."):
            return path[6:]
        if path.startswith("language_model.model."):
            return path[21:]
        return path

    for k in keyset:
        if not k.endswith('.scales'):
            continue
        base = k[:-7]  # Remove .scales
        weight_key = f"{base}.weight"
        if weight_key not in weights:
            continue
        
        # Get shapes
        G = int(weights[k].shape[1])
        P = int(weights[weight_key].shape[1])
        
        # Try different bit widths
        for b in [6, 4, 3, 2, 8]:
            for gs in [64, 128, 32, 16, 8]:
                if P * 32 == G * gs * b:
                    canon_path = _canon(base)
                    per_module_params[canon_path] = {
                        "group_size": gs,
                        "bits": b
                    }
                    if DEBUG >= 6:
                        print(f"[quantize] Pre-computed for {canon_path}: bits={b}, group_size={gs}")
                    break
except Exception as e:
    if DEBUG >= 2:
        print(f"[quantize] per-module params precompute failed: {e}")
```

さらに495行目付近の`derive_params_from_weights`関数を修正：

```python
def derive_params_from_weights(path: str):
    # ... 既存のコード ...
    
    # bit_candidates の行を以下に変更
    bit_candidates = [6, 4, 3, 2, 8]  # 6を最初に追加
    
    # ... 残りのコード ...
```

#### 3.2 main.py の修正

`exo/exo/main.py` の240行目付近を確認し、以下のコードが存在することを確認：

```python
# Preemptively load the shard for this node when a prompt starts anywhere
def preemptively_load_shard(request_id: str, opaque_status: str):
    try:
        status = json.loads(opaque_status)
        if status.get("type") != "node_status" or status.get("status") != "start_process_prompt":
            return
        base = status.get("base_shard") or status.get("shard")
        if not base:
            return
        base_shard = Shard.from_dict(base)
        current_shard = node.get_current_shard(base_shard)
        if DEBUG >= 2:
            print(f"Preemptively starting ensure_shard for {current_shard}")
        asyncio.create_task(node.inference_engine.ensure_shard(current_shard))
    except Exception as e:
        if DEBUG >= 2:
            print(f"Failed to preemptively start ensure_shard: {e}")
            traceback.print_exc()

node.on_opaque_status.register("preemptively_load_shard").on_next(preemptively_load_shard)
```

もし存在しない場合は追加してください。

### 4. モデル設定の追加

`exo/exo/models.py` に以下を追加：

```python
"kimi-k2-4bit": {
    "MLXDynamicShardInferenceEngine": {
        "repo": "inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit",
        "local_path": Path.home() / "models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit"
    },
    "layers": 61
}
```

### 5. モデルconfig.jsonの修正

`~/models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit/config.json` を編集：

```json
{
  "model_type": "deepseek_v3",  // kimi_k2 から変更
  // ... 他の設定はそのまま
}
```

## 🚀 起動手順

### 1. ネットワーク設定

#### Thunderbolt Bridge の設定
1. 両方のMacをThunderbolt 4ケーブルで接続
2. システム設定 → ネットワーク → Thunderbolt Bridge
3. IPアドレスを手動設定：
   - M2側: 10.0.2.1/24
   - M3側: 10.0.2.2/24

### 2. Discovery設定ファイルの作成

両方のMacで `/Users/{username}/exo/discovery.json` を作成：

```json
{
  "peers": {
    "m2-studio-main": {
      "address": "10.0.2.1",
      "port": 50051,
      "device_capabilities": { 
        "model": "Mac Studio",
        "chip": "Apple M2 Ultra",
        "memory": 196608,
        "flops": { "fp32": 26.98, "fp16": 53.96, "int8": 107.92 }
      }
    },
    "m3-studio-worker": {
      "address": "10.0.2.2",
      "port": 50052,
      "device_capabilities": {
        "model": "Mac Studio",
        "chip": "Apple M3 Ultra",
        "memory": 524288,
        "flops": { "fp32": 54.26, "fp16": 108.52, "int8": 217.04 }
      }
    }
  }
}
```

### 3. ノードの起動

#### M3側（Worker）を先に起動

```bash
cd ~/exo/exo
source venv/bin/activate

DEBUG=3 HF_HUB_DISABLE_TELEMETRY=1 python -m exo.main \
  --disable-tui \
  --inference-engine mlx \
  --default-model kimi-k2-4bit \
  --node-id m3-studio-worker \
  --node-host 10.0.2.2 \
  --node-port 50052 \
  --discovery-module manual \
  --discovery-config-path ~/exo/discovery.json \
  --wait-for-peers 0
```

#### M2側（Main）を起動

```bash
cd ~/exo/exo
source venv/bin/activate

DEBUG=3 HF_HUB_DISABLE_TELEMETRY=1 python -m exo.main \
  --disable-tui \
  --inference-engine mlx \
  --default-model kimi-k2-4bit \
  --node-id m2-studio-main \
  --node-host 10.0.2.1 \
  --node-port 50051 \
  --discovery-module manual \
  --discovery-config-path ~/exo/discovery.json \
  --wait-for-peers 1
```

## 📝 推論の実行

### APIエンドポイント経由

```bash
curl -X POST http://10.0.2.1:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-4bit",
    "messages": [
      {"role": "user", "content": "Hello, tell me about artificial intelligence"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### コマンドライン経由

```bash
DEBUG=3 python -m exo.main run kimi-k2-4bit \
  --prompt "Hello, how are you?" \
  --disable-tui \
  --node-id m2-studio-main \
  --node-host 10.0.2.1 \
  --node-port 50051 \
  --discovery-module manual \
  --discovery-config-path ~/exo/discovery.json \
  --wait-for-peers 1
```

## 🔍 トラブルシューティング

### 1. メモリ不足エラー

**症状**: "ValueError: Expected shape (163840, 672) but received shape (163840, 1344)"

**解決策**: sharded_utils.pyのパッチが正しく適用されているか確認。特に層別量子化の検出ロジック。

### 2. トークナイザーエラー

**症状**: "No chat template defined"

**解決策**: 
```bash
pip install tiktoken blobfile
```

### 3. 接続エラー

**症状**: ノード間で通信できない

**解決策**:
- Thunderbolt Bridgeの設定確認
- ファイアウォールの無効化
- `ping 10.0.2.1` / `ping 10.0.2.2` で疎通確認

### 4. モデルロードが遅い

**症状**: 初回起動時に長時間かかる

**対策**:
- 初回は最大30分程度かかる場合がある
- 2回目以降はキャッシュにより高速化
- Activity MonitorでPython プロセスのメモリ使用量を監視

## 📊 期待される動作

### メモリ使用量
- **M2側**: 約145-150GB（レイヤー44-60を担当）
- **M3側**: 約360-370GB（レイヤー0-43を担当）

### パフォーマンス
- 初回トークン生成: 約50-60秒
- 以降のトークン: 約5-10秒/トークン
- メモリスワップなしで動作

## 🎯 成功の確認

以下が確認できれば成功です：

1. 両ノードでメモリが正しく分散ロード
2. トークンが継続的に生成される
3. エラーなく推論が完了する

## 📚 技術詳細

### 層別量子化の仕組み
- **embed_tokens**: 3bit量子化（最も圧縮）
- **attention層 (q/k/v)**: 4bit量子化（標準）
- **lm_head**: 6bit量子化（精度重視）

### メモリ分割戦略
- レイヤー数ベースではなくメモリサイズベースで分割
- M3により多くのレイヤーを割り当て（メモリ容量に比例）

## 🤝 コントリビューション

このガイドの改善や問題報告は以下へ：
- GitHub Issues: [あなたのリポジトリURL]
- 改善提案はPull Requestでお願いします

## 📄 ライセンス

このガイドはMITライセンスで公開されています。
モデル自体のライセンスはMoonshot AIの規約に従ってください。

---

*Last updated: 2025-09-01*
*Successfully tested with Kimi-K2 1T model on M2 Studio + M3 Ultra*