# Local RLHF

[English](README.md) | [简体中文](README.zh-CN.md)

完整的 Apple Silicon 本地 RLHF 训练工具，支持 SFT、Reward Model 和 PPO，并通过 LoRA 降低训练内存占用。

安装方法与完整快速入门见[仓库中文 README](../../README.zh-CN.md)。激活环境后，在当前目录执行：

```bash
python train_rlhf.py --device mps
local-rlhf chat --model output/local-rlhf/ppo/actor
```

| 路径 | 用途 |
| --- | --- |
| `local_rlhf/` | CLI、分词、模型、SFT/RM/DPO 训练器与经典 PPO |
| `train_rlhf.py` | 依次启动 SFT → Reward Model → PPO |
| `configs/` | M1 16GB 和更小模型的默认配置 |
| `data/` | 少量演示数据 |
| `tests/` | 目标函数、掩码、冻结参考模型及完整流程测试 |
| `scripts/smoke_test.py` | 无需下载模型的 MPS/CPU 全流程测试 |
| `examples/chat.py` | 加载训练检查点的交互式／单次聊天示例 |
| `docs/` | 训练说明、数据格式和验证结果 |

使用 `local-rlhf --help` 或 `local-rlhf ppo --help` 查看参数。
