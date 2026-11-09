"""SFT/DPO sequence likelihoods and the classic clipped PPO objectives."""

import torch
import torch.nn.functional as F


def token_log_probs(logits, ids):
    # Keep the causal shift and selected-token probabilities explicit for PPO/DPO.
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    return log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def response_log_probs(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   use_cache=False).logits
    mask = batch["labels"][:, 1:].ne(-100)
    return (token_log_probs(logits, batch["input_ids"]) * mask).sum(dim=-1)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected))
    return -F.logsigmoid(margin).mean()


def generalized_advantages(rewards, values, gamma=1.0, lam=0.95):
    """A rollout is a finite episode; EOS and the generation cap are terminal."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[-1])
    next_value = torch.zeros_like(values[-1])
    for t in reversed(range(rewards.shape[-1])):
        delta = rewards[t] + gamma * next_value - values[t]
        carry = delta + gamma * lam * carry
        advantages[t] = carry
        next_value = values[t]
    return advantages, advantages + values


def clipped_policy_loss(log_probs, old_log_probs, advantages, clip_range):
    ratio = (log_probs - old_log_probs).exp()
    return torch.maximum(-advantages * ratio,
                         -advantages * ratio.clamp(1 - clip_range, 1 + clip_range)).mean()


def clipped_value_loss(values, old_values, returns, clip_range):
    clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
    return 0.5 * torch.maximum((values - returns).square(), (clipped - returns).square()).mean()
