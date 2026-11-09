"""Supervised, pairwise reward and optional DPO training."""

import torch
import torch.nn.functional as F

from .data import collate, prepare_rows, read_jsonl
from .models import (load_policy, load_reward, load_tokenizer, reference_policy,
                     save_policy, save_reward, select_device, snapshot_reference)
from .objectives import dpo_loss, response_log_probs
from .runtime import (Run, backward, batches, groups, optimizer_step, seed_everything,
                      trainable_parameters, validate_output)


def loss_weight(stage, rows):
    # Causal LM loss is averaged over response tokens, preference loss over pairs.
    if stage == "sft":
        return sum(sum(label != -100 for label in row["labels"][1:]) for row in rows)
    return len(rows)


def compute_loss(stage, model, rows, tokenizer, device, beta):
    if stage == "sft":
        batch = collate(rows, tokenizer.pad_token_id, device)
        return model(**batch, use_cache=False).loss, {}
    chosen = collate([row["chosen"] for row in rows], tokenizer.pad_token_id, device)
    rejected = collate([row["rejected"] for row in rows], tokenizer.pad_token_id, device)
    if stage == "reward":
        chosen_score = model(chosen["input_ids"], chosen["attention_mask"])
        rejected_score = model(rejected["input_ids"], rejected["attention_mask"])
        margin = chosen_score - rejected_score
        return -F.logsigmoid(margin).mean(), {"preference_accuracy": float((margin > 0).float().mean())}
    with reference_policy(model):
        ref_chosen = response_log_probs(model, chosen)
        ref_rejected = response_log_probs(model, rejected)
    chosen_logp = response_log_probs(model, chosen)
    rejected_logp = response_log_probs(model, rejected)
    loss = dpo_loss(chosen_logp, rejected_logp, ref_chosen, ref_rejected, beta)
    margin = (chosen_logp - ref_chosen) - (rejected_logp - ref_rejected)
    return loss, {"preference_accuracy": float((margin > 0).float().mean())}


def train(args):
    validate_output(args.output)
    seed_everything(args.seed)
    device = select_device(args.device)
    tokenizer = load_tokenizer(args.model, args.revision)
    rows = prepare_rows(read_jsonl(args.data, args.max_samples), tokenizer, args.command,
                        args.max_length, args.max_prompt_length)
    evaluation = None
    if args.eval_data:
        evaluation = prepare_rows(read_jsonl(args.eval_data, args.max_samples), tokenizer,
                                  args.command, args.max_length, args.max_prompt_length)
    loader = load_reward if args.command == "reward" else load_policy
    model = loader(args.model, device, args.lora_rank, args.gradient_checkpointing, revision=args.revision)
    config = model.policy.config if args.command == "reward" else model.config
    if args.max_length > config.max_position_embeddings:
        raise ValueError("max_length exceeds the model context window")
    if args.command == "dpo":
        snapshot_reference(model)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
                                 foreach=False)
    run = Run(args, device)
    print(f"{args.command}: {device}, {len(rows)} examples, "
          f"{sum(p.numel() for p in parameters):,} trainable parameters", flush=True)
    for epoch in range(args.epochs):
        model.train()
        for group in groups(batches(rows, args.batch_size, args.seed, epoch), args.gradient_accumulation_steps):
            # Preserve the objective across variable sequence lengths and partial groups.
            normalizer = sum(loss_weight(args.command, batch_rows) for batch_rows in group)
            loss_sum, metrics_sum = 0.0, {}
            for batch_rows in group:
                loss, metrics = compute_loss(args.command, model, batch_rows, tokenizer, device, args.beta)
                weight = loss_weight(args.command, batch_rows) / normalizer
                backward(loss * weight)
                loss_sum += float(loss.detach()) * weight
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value * weight
            grad_norm = optimizer_step(optimizer, parameters, args.max_grad_norm)
            run.steps += 1
            run.log(step=run.steps, epoch=epoch + 1, loss=loss_sum, grad_norm=grad_norm, **metrics_sum)
            if args.max_steps and run.steps >= args.max_steps:
                break
        if args.max_steps and run.steps >= args.max_steps:
            break
    eval_metrics = {}
    if evaluation:
        model.eval()
        normalizer = loss_weight(args.command, evaluation)
        with torch.no_grad():
            for batch_rows in batches(evaluation, args.batch_size, args.seed, 0):
                loss, metrics = compute_loss(args.command, model, batch_rows, tokenizer, device, args.beta)
                for key, value in {"loss": float(loss), **metrics}.items():
                    weight = loss_weight(args.command, batch_rows) / normalizer
                    eval_metrics[key] = eval_metrics.get(key, 0.0) + value * weight
        run.log(evaluation=eval_metrics)
    if args.command == "reward":
        save_reward(model, tokenizer, run.output)
    else:
        save_policy(model, tokenizer, run.output)
    return run.finish(examples=len(rows), evaluation=eval_metrics)
