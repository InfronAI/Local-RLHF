# Local RLHF

[English](README.md) | [简体中文](README.zh-CN.md)

A complete local RLHF training toolkit for Apple Silicon, featuring supervised fine-tuning, reward modeling, and PPO with memory-efficient LoRA.

See the [repository README](../../README.md) for installation and the complete quick start.
From this directory, with the environment activated:

```bash
python train_rlhf.py --device mps
local-rlhf chat --model output/local-rlhf/ppo/actor
```

| Path | Purpose |
| --- | --- |
| `local_rlhf/` | CLI, tokenization, models, SFT/RM/DPO trainers and classic PPO |
| `train_rlhf.py` | Sequential SFT → Reward Model → PPO launcher |
| `configs/` | M1 16GB and smaller-model defaults |
| `data/` | Small demonstration datasets |
| `tests/` | Objective, masking, frozen-reference and training integration tests |
| `scripts/smoke_test.py` | Offline end-to-end test on MPS or CPU |
| `examples/chat.py` | Standalone interactive/one-shot chat with a trained checkpoint |
| `docs/` | Architecture, dataset format and validation evidence |

Use `local-rlhf --help` or `local-rlhf ppo --help` for command-line options.
