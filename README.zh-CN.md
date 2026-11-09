# Local RLHF

[English](README.md) | [简体中文](README.zh-CN.md)

A complete local RLHF training toolkit for Apple Silicon, featuring supervised fine-tuning, reward modeling, and PPO with memory-efficient LoRA.

面向 Apple Silicon 的本地 RLHF 训练工具，完整保留 **SFT → Reward Model → PPO** 三阶段流程。通过 PyTorch MPS、FP32 和 LoRA 在 Mac 上完成训练，也支持 CPU。默认使用 **Qwen2.5-0.5B-Instruct**，面向 **M1、16GB 统一内存** 配置，另提供可选 DPO 入口。

## 保留的训练流程

| 阶段 | 实现 |
| --- | --- |
| SFT | 因果语言模型训练，只计算回答部分的损失 |
| Reward Model | chosen/rejected 成对偏好、可训练标量头、成对 logistic loss |
| PPO | Actor、冻结 Reference、冻结 Reward Model，以及从奖励模型初始化并独立训练的 Critic |
| PPO 目标 | 逐 token KL 惩罚、终止奖励、GAE、策略裁剪、价值函数裁剪 |
| DPO | 相对于训练开始时冻结策略快照的直接偏好优化 |
| 推理 | 加载已保存的适配器，支持交互聊天和单次回答 |

Actor 与 Reference 共用一个冻结底座，使用独立 LoRA；Critic 与 Reward 共用另一个冻结底座，使用独立 LoRA 和标量头。**可训练参数相互独立**，四个 PPO 角色完整保留，同时减少底座副本带来的内存占用。

## 快速开始

使用原生 **arm64 Python 3.11 或更新版本**及支持 MPS 的 macOS。本项目实际验证使用 Python 3.11 和 PyTorch 2.15.0。

```bash
git clone https://github.com/InfronAI/Local-RLHF.git
cd Local-RLHF
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e './applications/local-rlhf[test]'
local-rlhf doctor
```

确认输出中 `mps_available` 为 `true`，运行完整三阶段训练：

```bash
cd applications/local-rlhf
python train_rlhf.py --device mps
```

首次运行从 Hugging Face 下载底座权重，训练在本地完成。内置的每类八条样例仅用于演示流程，**不足以训练出高质量助手，也不能据此证明模型能力提升**。

模型完整缓存后，可使用 `HF_HUB_OFFLINE=1 python train_rlhf.py --device mps` 跳过网络元数据检查。再次训练时，请通过 `--output` 使用新的输出目录。

加载 PPO 训练后的 Actor：

```bash
local-rlhf chat --model output/local-rlhf/ppo/actor --device mps
local-rlhf chat --model output/local-rlhf/ppo/actor --prompt "请解释什么是 LoRA。"
# 独立示例脚本，支持相同参数
python examples/chat.py --model output/local-rlhf/ppo/actor --device mps
```

## 单独运行各阶段

以下命令均在 `applications/local-rlhf` 目录中运行，并保持虚拟环境激活：

```bash
local-rlhf sft --config configs/m1_16gb.json \
  --data data/sft.jsonl --output output/sft

local-rlhf reward --config configs/m1_16gb.json \
  --data data/preferences.jsonl --output output/reward

local-rlhf ppo --config configs/m1_16gb.json \
  --model output/sft --reward-model output/reward \
  --data data/prompts.jsonl --output output/ppo

# 可选：在 SFT 后运行 DPO
local-rlhf dpo --config configs/m1_16gb.json \
  --model output/sft --data data/preferences.jsonl --output output/dpo
```

`python -m local_rlhf` 与 `local-rlhf` 等价。显式命令行参数优先于 JSON 配置。输出目录必须不存在或为空，避免覆盖已有结果。

## 16GB 默认配置

| 配置项 | 默认值 |
| --- | --- |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 设备 / 精度 | 优先 MPS / FP32 |
| LoRA rank / 目标层 | 8 / 注意力 `q_proj`、`v_proj` |
| Microbatch / 梯度累积 | 1 / 8 |
| 序列 / Prompt / 生成长度 | 256 / 128 / 64 tokens |
| 激活检查点 | 开启 |
| 训练限制 | 1 个 epoch，每阶段最多 100 次优化器更新 |
| 每组 rollout 的 PPO 更新次数 | 2 |

内置八条样例按默认配置产生 1 次 SFT 更新、1 次奖励模型更新和 2 次 PPO 更新。`--max-steps` 表示上限，不会自动重复数据；延长训练请增加 `--epochs` 并使用合适的数据。

若需更低内存，可使用 `--config configs/small.json` 切换到偏英文的 `SmolLM2-135M-Instruct`。实际占用取决于序列长度、模型大小及其他应用。内存压力较高时，缩短序列、降低 PPO rollout 累积数量，或使用更小模型；不要关闭 MPS 内存安全限制。

## 不下载模型的验证

```bash
# 在仓库根目录执行
pytest applications/local-rlhf/tests -q
python applications/local-rlhf/scripts/smoke_test.py \
  --device mps --output applications/local-rlhf/output/smoke
```

脚本在本地创建随机微型模型，依次验证 SFT、Reward Model、PPO、DPO，以及检查点重载推理。没有 MPS 时改用 `--device cpu`。这类测试验证流程正确性，不衡量语言能力。

## 文档与来源

- [中文训练说明](applications/local-rlhf/docs/training.zh-CN.md)
- [训练架构与检查点说明（英文）](applications/local-rlhf/docs/training.md)
- [数据格式与校验（英文）](applications/local-rlhf/docs/data.md)
- [实际验证结果与限制（英文）](applications/local-rlhf/docs/validation.md)
- [真实模型 CPU/MPS 梯度对比脚本](applications/local-rlhf/scripts/check_mps_gradients.py)

所有实现代码位于 `applications/local-rlhf`；模型、虚拟环境、缓存与日志不纳入 Git。

本项目基于 [DeepSpeed-Chat](https://github.com/deepspeedai/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) 的训练流程重写，已移除原有 CUDA、DeepSpeed、多卡训练及其他示例代码。新实现使用目标仓库原有的 [MIT 许可证](LICENSE)，同时保留上游 [Apache-2.0 许可证](LICENSE.upstream)和[来源说明](NOTICE)。模型权重遵循各自模型卡中的许可证。
