# Validation record

Validation date: **2026-11-09**. These results describe functional training checks, not a model-quality benchmark.

## Hardware and environment

- Apple M1 Pro, 16GB unified memory, native arm64 Python 3.11.16.
- macOS 26.6.1; PyTorch 2.15.0 with MPS available.
- Transformers 4.56.2, PEFT 0.17.1, Accelerate 1.10.1.
- FP32, LoRA, activation checkpointing; no CUDA or DeepSpeed runtime.

## Correctness tests

**12 tests passed** on CPU in the final correctness run.

The test suite covers response-only labels, multi-turn masking, EOS/PAD handling, bounded JSONL reads, rejected invalid preferences, causal log-probability alignment, DPO gradients, GAE terminal returns, clipped PPO policy/value losses, frozen reference/base parameters, reward checkpoint reload, full SFT/RM/PPO/DPO training, independently updated actor/critic parameters, multiple rollout groups, partial gradient accumulation, config precedence and output overwrite protection.

```bash
pytest applications/local-rlhf/tests -q
```

The MPS smoke test creates a random tiny Llama model and tokenizer without downloading weights. It completed SFT, reward modeling, PPO, DPO and saved-actor generation. It exercises checkpointing, actual gradients, optimizers, serialization and reloading.

```bash
python applications/local-rlhf/scripts/smoke_test.py \
  --device mps --output applications/local-rlhf/output/smoke
```

## Real-model gradient comparison

The final dependencies pin PyTorch **2.15.0**. During development, PyTorch 2.8.0 on this Mac produced matching forward losses but incorrect Qwen gradients for some sequence lengths. Tiny-model tests and finite losses alone did not expose this. The implementation therefore also includes a real-model CPU/MPS backward comparison.

Eight preference pairs with different chosen/rejected lengths passed the comparison on 2.15.0: gradient norm ratios were within 0.005% of one and gradient cosine similarity exceeded 0.99999999. This is an empirical check of this model, dataset and environment, not a guarantee for every MPS model or system.

```bash
HF_HUB_OFFLINE=1 python applications/local-rlhf/scripts/check_mps_gradients.py \
  --model applications/local-rlhf/output/verified-mps/sft \
  --output applications/local-rlhf/output/mps-gradients.json
```

The script compares against CPU FP32, exits with an error on a mismatch and saves its measurements. See the captured [gradient comparison results](gradients.json). Preserve the tested PyTorch pin when reproducing this environment.

## Default Qwen model on MPS

The complete default pipeline ran with `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, the included eight-example datasets, and `configs/m1_16gb.json`.

```bash
HF_HUB_OFFLINE=1 python applications/local-rlhf/train_rlhf.py --device mps \
  --output applications/local-rlhf/output/verified-mps
```

| Stage | Optimizer updates | Recorded stage time | Highest sampled MPS driver memory |
| --- | ---: | ---: | ---: |
| SFT | 1 | 3.44 s | 2.62 GiB |
| Reward Model | 1 | 2.66 s | 2.61 GiB |
| PPO | 2 | 27.60 s | 4.70 GiB |
| DPO (optional) | 1 | 3.66 s | 2.64 GiB |

Timing begins after model/data loading and includes final checkpoint writing. It excludes the first weight download, model construction, tokenizer loading, process startup and network metadata requests. Memory is sampled at logged optimizer updates; these are **not true peak-memory measurements**. PyTorch MPS driver memory and process RSS should not be added as independent unified-memory allocations.

PPO collected eight responses totaling 453 response tokens, then updated the actor and critic twice. Training losses and gradients were finite. The saved PPO actor was reloaded and generated an English response through `examples/chat.py` on MPS.

The default SFT and reward runs are tiny, one-update demonstrations. Reward training accuracy is measured on the training examples. No held-out quality, reward-model calibration, long-run stability, or task-level improvement is established. The chat check verifies loading and execution only; responses were not scored for factual accuracy.

Raw stage summaries are recorded in [validation.json](validation.json). Model weights and generated output directories are intentionally excluded from version control.

## Scope

The hardware run used an **M1 Pro**, not every M1 variant or macOS version. The smaller configuration and CPU path provide alternatives; neither guarantees a particular memory budget or throughput on a different machine. LoRA training is implemented for compatible Hugging Face causal LMs with `q_proj`/`v_proj` layers. The Qwen default and tiny Llama test model were directly verified.
