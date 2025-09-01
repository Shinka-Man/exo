# exo - Distributed Inference for 1T Parameter Models

🎉 **World's First**: Successfully running Kimi-K2 1 Trillion Parameter Model on personal hardware!

## 📺 What's This?

This is a fork of [exo-explore/exo](https://github.com/exo-explore/exo) with critical patches to enable distributed inference of massive models like Kimi-K2 (1T parameters, 504GB) across multiple Apple Silicon Macs.

## 🚀 Key Features

- **Layer-wise Mixed Quantization**: Supports different bit widths per layer (3/4/6bit)
- **Distributed Memory Loading**: Splits 504GB model across M2 Studio (192GB) + M3 Ultra (512GB)
- **MoE Optimization**: Efficiently handles 384 experts with Top-8 routing
- **Thunderbolt Network**: Direct Mac-to-Mac connection for minimal latency

## 📊 Achievement

- **Model**: Kimi-K2-Instruct-MLX-3.985bit (1 Trillion Parameters)
- **Hardware**: M2 Studio + M3 Ultra via Thunderbolt 4
- **Memory Usage**: M2: ~145GB, M3: ~360GB
- **Performance**: 0.076 TPS (with potential for 10 TPS after optimization)

## 📚 Documentation

- [Complete Setup Guide](Kimi-K2_分散推論_完全ガイド.md) - Step-by-step reproduction guide
- [Success Snapshot](Kimi-K2_成功スナップショット_20250901.md) - Current working state
- [Technical Summary](Kimi-K2_分散推論チャレンジ総括.md) - Project overview

## 🔧 Critical Patches

### 1. `exo/inference/mlx/sharded_utils.py`
- Automatic detection of per-layer quantization parameters
- Support for mixed bit-width (3/4/6bit) in single model

### 2. `exo/main.py`
- Preemptive shard loading for simultaneous memory allocation
- Distributed loading synchronization

### 3. Model Configuration
- DeepSeek-V3 compatibility for Kimi-K2
- Proper MoE layer handling

## 🛠️ Quick Start

1. **Setup Two Macs**
   ```bash
   # Clone this repo on both machines
   git clone https://github.com/Shinka-Man/exo.git
   cd exo
   
   # Install dependencies
   python -m venv venv
   source venv/bin/activate
   pip install -e .
   pip install tiktoken blobfile
   ```

2. **Configure Network**
   - Connect Macs via Thunderbolt 4
   - Set static IPs (10.0.2.1 and 10.0.2.2)
   - Edit `discovery.json` with node details

3. **Run Distributed Inference**
   ```bash
   # On M3 (Worker)
   python -m exo.main --node-id m3-studio-worker ...
   
   # On M2 (Main)
   python -m exo.main --node-id m2-studio-main ...
   ```

## 🎯 Performance Optimization Roadmap

Current: 0.076 TPS → Target: 10 TPS

- [ ] Expert parallelization
- [ ] Communication optimization
- [ ] Metal Performance Shaders integration
- [ ] Dynamic batching

## 🤝 Contributing

This is a research project pushing the boundaries of what's possible with personal hardware. Contributions for performance improvements are welcome!

## 📄 License

MIT License - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- Original [exo](https://github.com/exo-explore/exo) team
- Moonshot AI for Kimi-K2 model
- Apple for incredible M-series chips

---

*"Making the impossible possible - 1 trillion parameters on your desk"*

**Date**: September 1, 2025  
**Location**: Tokyo, Japan  
**Hardware**: M2 Studio (192GB) + M3 Ultra (512GB)