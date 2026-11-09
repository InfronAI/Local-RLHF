"""One CLI for the local training pipeline, environment checks and inference."""

import argparse
import json
import math
import platform
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def add_training_arguments(parser):
    parser.add_argument("--config", help="JSON training defaults; explicit CLI flags take precedence")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID, local model or local LoRA checkpoint")
    parser.add_argument("--revision", help="Optional base model commit/revision")
    parser.add_argument("--data", required=True, help="Local JSONL file")
    parser.add_argument("--eval-data", help="Optional separate validation JSONL (SFT/reward/DPO)")
    parser.add_argument("--output", required=True, help="New/empty checkpoint directory")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100, help="Optimizer update cap; 0 uses all epochs")
    parser.add_argument("--max-samples", type=int, default=10000, help="Input row cap; 0 reads all rows")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local RLHF: MPS/CPU LoRA training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {}
    for name in ["sft", "reward", "ppo", "dpo"]:
        commands[name] = subparsers.add_parser(name)
        add_training_arguments(commands[name])
    ppo = commands["ppo"]
    ppo.add_argument("--reward-model", required=True)
    ppo.add_argument("--ppo-epochs", type=int, default=2)
    ppo.add_argument("--critic-learning-rate", type=float, default=5e-5)
    ppo.add_argument("--kl-coef", type=float, default=0.1)
    ppo.add_argument("--clip-range", type=float, default=0.2)
    ppo.add_argument("--value-clip-range", type=float, default=0.2)
    ppo.add_argument("--value-loss-coef", type=float, default=0.5)
    ppo.add_argument("--reward-clip", type=float, default=5.0)
    ppo.add_argument("--gamma", type=float, default=1.0)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    chat = subparsers.add_parser("chat")
    chat.add_argument("--model", required=True)
    chat.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    chat.add_argument("--prompt", help="One-shot generation; omit for interactive chat")
    chat.add_argument("--max-prompt-length", type=int, default=256)
    chat.add_argument("--max-new-tokens", type=int, default=128)
    subparsers.add_parser("doctor", help="Report Python, architecture, memory and MPS availability")
    args = parser.parse_args(argv)
    if args.command in commands and args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            parser.error("config must be a JSON object")
        actions = {action.dest: action for action in commands[args.command]._actions
                   if not action.required and action.dest not in {"help", "config", "eval_data"}}
        for key, value in config.items():
            if key not in actions:
                parser.error(f"Unknown/unsupported config key: {key}")
            action = actions[key]
            expected = action.type or (bool if isinstance(action.default, bool) else str)
            if (expected is int and type(value) is not int
                    or expected is float and (type(value) not in {int, float} or not math.isfinite(value))
                    or expected is bool and type(value) is not bool
                    or expected is str and not isinstance(value, str)):
                parser.error(f"Invalid config type for {key}: expected {expected.__name__}")
            if action.choices and value not in action.choices:
                parser.error(f"Invalid config choice for {key}: {value}")
        commands[args.command].set_defaults(**config)
        args = parser.parse_args(argv)
    if args.command in commands:
        positive = ["epochs", "batch_size", "gradient_accumulation_steps", "max_length",
                    "max_prompt_length", "max_new_tokens", "lora_rank", "learning_rate", "max_grad_norm", "beta"]
        if args.command == "ppo":
            positive += ["ppo_epochs", "critic_learning_rate", "clip_range", "value_clip_range",
                         "value_loss_coef", "reward_clip"]
        for key in positive:
            value = getattr(args, key)
            if not math.isfinite(value) or value <= 0:
                parser.error(f"--{key.replace('_', '-')} must be finite and positive")
        for key in ["max_steps", "max_samples", "weight_decay"]:
            if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
                parser.error(f"--{key.replace('_', '-')} must be nonnegative")
        if args.max_prompt_length > args.max_length - 2:
            parser.error("max_prompt_length must be at most max_length - 2")
        if args.command == "ppo":
            if args.max_prompt_length + args.max_new_tokens > args.max_length:
                parser.error("PPO prompt + max_new_tokens must fit within max_length")
            if args.batch_size != 1:
                parser.error("Local PPO uses batch_size=1; increase gradient_accumulation_steps for more rollouts")
            if args.eval_data:
                parser.error("PPO does not use --eval-data; evaluate the saved actor with chat")
            if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1 or not 0 <= args.kl_coef < float('inf'):
                parser.error("gamma/gae_lambda must be in [0,1], kl_coef must be finite and nonnegative")
    if args.command == "chat" and (args.max_new_tokens <= 0 or args.max_prompt_length <= 0):
        parser.error("Chat token limits must be positive")
    return args


def doctor():
    import psutil
    import torch
    print(json.dumps({"python": platform.python_version(), "architecture": platform.machine(),
                      "system": platform.platform(), "torch": torch.__version__,
                      "memory_gib": round(psutil.virtual_memory().total / 1024 ** 3, 2),
                      "mps_built": torch.backends.mps.is_built(),
                      "mps_available": torch.backends.mps.is_available(),
                      "default_device": "mps" if torch.backends.mps.is_available() else "cpu"}, indent=2))


def chat(args):
    import torch
    from transformers import GenerationConfig
    from .data import prompt_ids
    from .models import load_policy, load_tokenizer, select_device
    if (Path(args.model) / "reward_head.safetensors").exists():
        raise ValueError("Use the SFT/DPO checkpoint or PPO actor, not a reward/critic checkpoint")
    device = select_device(args.device)
    tokenizer = load_tokenizer(args.model)
    model = load_policy(args.model, device, checkpointing=False, trainable=False)
    context = model.config.max_position_embeddings
    if args.max_prompt_length + args.max_new_tokens > context:
        raise ValueError("Chat prompt + generation limit exceeds model context window")
    history = []
    while True:
        if args.prompt is not None:
            prompt = args.prompt
        else:
            try:
                prompt = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue
        history.append({"role": "user", "content": prompt})
        ids = torch.tensor([prompt_ids(tokenizer, history, args.max_prompt_length)], device=device)
        with torch.no_grad():
            sequence = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                      generation_config=GenerationConfig(
                                          max_new_tokens=args.max_new_tokens, do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id,
                                          eos_token_id=tokenizer.eos_token_id, use_cache=True))
        answer = tokenizer.decode(sequence[0, ids.shape[1]:], skip_special_tokens=True)
        print(answer, flush=True)
        history.append({"role": "assistant", "content": answer})
        # Bound Python-side history too; actual token history is truncated above.
        history = history[-20:]
        if args.prompt is not None:
            break


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "chat":
            return chat(args)
        if args.command == "ppo":
            from .ppo import train_ppo
            train_ppo(args)
        else:
            from .trainers import train
            train(args)
    except (ValueError, FileNotFoundError, FloatingPointError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
