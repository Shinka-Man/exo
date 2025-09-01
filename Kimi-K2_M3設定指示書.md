# Kimi-K2 1Tモデル分散推論設定 - M3 Ultra側作業指示書

## 概要
Kimi-K2（1兆パラメータ、504GB）モデルを M2 + M3 Ultra で分散推論するための設定。
DeepSeek-R1ベースの61レイヤー構造を2台で分割処理。

## モデル情報
- **モデル名**: Kimi-K2-Instruct-MLX-3.985bit
- **総パラメータ数**: 1兆（アクティブ32B）
- **ファイルサイズ**: 504GB
- **レイヤー数**: 61層
- **コンテキスト長**: 131,072トークン（128K）
- **ベース**: DeepSeek-R1アーキテクチャ
- **量子化**: 3.985bit

## 作業内容

### 1. モデルファイルの配置確認

M3側にもモデルファイルが必要です。以下のいずれかの方法で準備：

**オプションA: シンボリックリンク（推奨、Thunderbolt接続済みの場合）**
```bash
# M2側のモデルをマウント経由で参照
ln -s /Volumes/M2-Studio/Users/lna/models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit \
      /Users/kitt/models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit
```

**オプションB: ローカルコピー（504GB必要）**
```bash
# 十分なストレージがある場合のみ
cp -r /Volumes/M2-Studio/Users/lna/models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit \
      /Users/kitt/models/inferencerlabs/
```

### 2. models.py の更新

**ファイル**: `/Users/kitt/exo/exo/exo/models.py`

#### 2.1 モデル定義の追加（95行目付近）

```python
# 以下の行を探す:
"deepseek-r1-3bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-3bit", }, },
### deepseek distills

# その間に以下を追加:
"deepseek-r1-3bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "mlx-community/DeepSeek-R1-3bit", }, },
### kimi-k2 (1T params, DeepSeek-R1 based)
"kimi-k2-4bit": { "layers": 61, "repo": { "MLXDynamicShardInferenceEngine": "/Users/kitt/models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit", }, },
### deepseek distills
```

**重要**: パスは `/Users/kitt/models/...` にすること（M2側は `/Users/lna/models/...`）

#### 2.2 表示名の追加（187行目付近）

```python
# 以下の行を探す:
"deepseek-r1-3bit": "Deepseek R1 (3-bit)",
"llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",

# その間に以下を追加:
"deepseek-r1-3bit": "Deepseek R1 (3-bit)",
"kimi-k2-4bit": "Kimi K2 1T (4-bit, 504GB)",
"llava-1.5-7b-hf": "LLaVa 1.5 7B (Vision Model)",
```

### 3. 起動手順

#### 3.1 既存プロセスの停止
```bash
pkill -f "exo.*--inference-engine mlx"
```

#### 3.2 M3側を先に起動
```bash
DEBUG=3 HF_HUB_DISABLE_TELEMETRY=1 /Users/kitt/exo/venv/bin/exo \
  --disable-tui \
  --inference-engine mlx \
  --default-model kimi-k2-4bit \
  --node-id m3-studio-worker \
  --node-host 10.0.2.2 \
  --node-port 50052 \
  --discovery-module manual \
  --discovery-config-path /Users/kitt/exo/discovery.json
```

**注意点**:
- `--default-model kimi-k2-4bit` に変更
- 初回起動時は大量のモデルファイルロードで時間がかかる（数分）

#### 3.3 起動確認
```bash
# 別ターミナルで確認
lsof -i :50052
# PythonプロセスがLISTEN状態になっていればOK
```

### 4. メモリ使用量の監視

起動後、以下でメモリ使用量を確認：
```bash
# プロセスのメモリ使用量
ps aux | grep -E "exo.*mlx" | grep -v grep | awk '{print $6/1024/1024 " GB"}'

# システム全体のメモリ状況
vm_stat | grep "Pages free\|Pages active\|Pages wired"
```

**期待される使用量**:
- M3側: 約250GB（レイヤー0-30を担当）
- M2側: 約250GB（レイヤー31-60を担当）

### 5. トラブルシューティング

#### エラー1: FileNotFoundError (モデルが見つからない)
→ モデルパスが正しいか確認（`/Users/kitt/models/...`）

#### エラー2: メモリ不足
→ 他のアプリを終了してメモリを確保

#### エラー3: ロード時間が長すぎる
→ 504GBのモデルなので初回は10-15分かかる可能性あり

### 6. M2側からのテスト

M3が正常起動したら、M2側から以下のコマンドでテスト：

```bash
# M2側で実行
curl -sS http://127.0.0.1:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kimi-k2-4bit",
    "messages": [
      {"role": "user", "content": "こんにちは！あなたはKimi K2モデルですか？"}
    ]
  }' -m 120
```

**注意**: 初回レスポンスは数分かかる可能性があります。

## 重要な注意事項

1. **ストレージ要件**
   - モデルファイル: 504GB
   - 実行時メモリ: 約250GB
   - 合計で750GB以上の空き容量が必要

2. **初回起動の遅延**
   - 504GBのモデルロードに10-15分かかる
   - 2回目以降はキャッシュが効いて高速化

3. **分散の仕組み**
   - M3: レイヤー0-30（前半）を処理
   - M2: レイヤー31-60（後半）を処理
   - Thunderbolt 4で高速通信

## 完了確認チェックリスト

- [ ] モデルファイルへのアクセス確認
- [ ] models.pyにkimi-k2-4bitを追加
- [ ] M3側でexoが正常起動
- [ ] メモリ使用量が適切（約250GB）
- [ ] M2からの接続確立
- [ ] テストリクエストに応答

問題が発生した場合は、具体的なエラーメッセージと共にM2側に連絡してください。