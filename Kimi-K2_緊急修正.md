# Kimi-K2 緊急修正指示

## M3側で必要な修正

### config.jsonの修正
```bash
# バックアップ作成
cp /Users/kitt/models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit/config.json \
   /Users/kitt/models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit/config.json.backup

# model_typeをdeepseek_v3に変更
sed -i '' 's/"model_type": "kimi_k2"/"model_type": "deepseek_v3"/' \
    /Users/kitt/models/inferencerlabs/Kimi-K2-Instruct-MLX-3.985bit/config.json
```

### exoプロセスの再起動
```bash
# 既存プロセスを停止
pkill -f "exo.*--inference-engine mlx"

# M3を再起動
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

## 理由
Kimi-K2はDeepSeek-R1ベースですが、config.jsonで`model_type: "kimi_k2"`となっているため、
MLXローダーが認識できません。`deepseek_v3`に変更することで正常に動作します。

M2側は既に修正済みです。