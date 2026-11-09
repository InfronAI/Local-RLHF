# 训练说明

[English](training.md) | [中文 README](../../../README.zh-CN.md)

## 三阶段完整流程

1. **SFT：**冻结底座，只训练注意力层的 LoRA。使用模型的聊天模板构造输入；Prompt 和历史对话的标签设为 `-100`，仅最后一轮回答和 EOS 参与交叉熵损失。
2. **Reward Model：**独立训练奖励模型的 LoRA 和标量头。取最后一个非 padding token 的分数，优化 `-logsigmoid(r(chosen) - r(rejected))`。EOS 与 PAD 使用相同 token ID 时，通过显式 attention mask 区分。
3. **PPO：**Actor 由 SFT 初始化，Critic 的 LoRA 与标量头由奖励模型初始化。复制 SFT 适配器作为冻结 Reference，复制奖励适配器及标量头作为冻结 Reward，之后分别优化 Actor 与 Critic。

`train_rlhf.py` 默认依次启动以上三个阶段。每个阶段使用独立进程，完成后释放内存。DPO 是独立的可选入口，不替代默认 PPO 流程。

## 四个角色如何节省内存

Actor / Reference 共用一个冻结 Transformer 底座，使用两个独立 LoRA；Critic / Reward 共用另一个冻结底座，使用两个独立 LoRA 和两个独立标量头。Reference 是训练后的 SFT 快照，而不是没有 SFT 适配器的原始模型。冻结角色不进入优化器，Actor 与 Critic 的可训练参数相互独立。

PPO 每次逐条生成 `gradient_accumulation_steps` 个回答，构成一组 rollout，再进行 `ppo_epochs` 次更新。采样固定 temperature=1、top-p=1、top-k=0，保证采样策略与记录的 log probability 一致；关闭 dropout，避免新旧策略比率被随机扰动。

奖励由每个回答 token 的 KL 惩罚以及末尾的奖励模型分数组成。使用 GAE 计算 advantage 与 return，策略采用 PPO clipped surrogate，Critic 采用裁剪后的价值损失。EOS 和最大生成长度都被当作当前有限 episode 的终止，末尾 bootstrap value 为零。Actor 与 Critic 使用各自的 AdamW 优化器及梯度裁剪。

`sampled_kl` 是采样 token 上的旧策略与 Reference log probability 差，样本均值可能略小于零。同一组 rollout 的多个 PPO epoch 共用该记录；首组 rollout 的值为零是正常现象，因为 Actor 与 Reference 初始相同。优势归一化后，初次 Actor loss 接近零也不代表梯度为零。

## M1 16GB 配置

默认模型为 `Qwen/Qwen2.5-0.5B-Instruct`，FP32，LoRA rank 8，目标层为 `q_proj` 和 `v_proj`；microbatch=1，梯度累积=8，最大序列长度=256，Prompt=128，生成=64，并开启非重入激活检查点。

训练不需要 DeepSpeed、CUDA、NCCL、bitsandbytes 或多卡进程。包在导入 PyTorch 前设置 MPS 算子 CPU fallback；若在其他程序已经导入 PyTorch 后再导入本包，请提前设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`。

PyTorch 固定为 2.15.0。开发中发现本机 2.8.0 对部分真实模型序列的反向梯度存在差异，因此增加了真实 Qwen 模型的 CPU/MPS 梯度对比脚本 `scripts/check_mps_gradients.py`，并在 2.15.0 上通过 8 组样本验证。复现实验时请保留该版本，不能只根据 loss 有限判断训练正确。

`--max-samples` 默认最多读取 10,000 条数据；设为 0 取消上限，但本实现会将选中的分词结果保存在内存中。大数据集应先准备有界本地子集。

## 数据和训练时长

命令行参数覆盖 JSON 默认值。`epochs` 表示数据集遍历次数，`max_steps` 是优化器更新次数上限，不会强制训练到指定次数。PPO 每次 Actor/Critic 更新对计作一步。

默认内置八条样例只产生 1 次 SFT、1 次奖励更新及 2 次 PPO 更新，用于验证流程。正式训练应换用足够规模且质量合适的数据，并提供独立验证集。SFT、RM、DPO 支持 `--eval-data`，训练结束后记录验证损失和偏好准确率。

```bash
python train_rlhf.py --device mps --epochs 3 --max-steps 0 \
  --sft-data /path/to/sft.jsonl \
  --preference-data /path/to/preferences.jsonl \
  --prompt-data /path/to/prompts.jsonl \
  --output output/my-training
```

## 产物和限制

每个阶段保存 `run.json`、`metrics.jsonl`、`summary.json`。SFT、DPO 保存 LoRA 和 tokenizer；RM 额外保存 `reward_head.safetensors`；PPO 保存 `actor/` 与 `critic/`，聊天时使用 `actor/`。

检查点仅包含适配器，不包含完整底座。可用时会记录 Hugging Face 底座的实际 revision。离线使用时必须保留底座缓存；若底座来自本地目录，该目录需继续存在。

可以把已保存适配器传给 `--model` 继续训练，但会创建新优化器，**不等同于精确断点续训**。本实现不保存优化器、随机数状态和 rollout buffer，不提供定期检查点或学习率调度。最终检查点在成功完成时写入；中途中断可能只留下日志。每次运行请使用新的输出目录。

模型质量、长时间训练稳定性和泛化效果需要在真实任务数据上单独评估。[验证记录](validation.md)仅说明已执行的功能与硬件测试。
