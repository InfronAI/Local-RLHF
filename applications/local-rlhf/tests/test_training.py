import json

import pytest
import torch
from safetensors.torch import load_file

from local_rlhf.cli import parse_args
from local_rlhf.ppo import train_ppo
from local_rlhf.trainers import train


def arguments(stage, model, data, output, extra=()):
    return parse_args([stage, "--model", str(model), "--data", str(data), "--output", str(output),
                       "--device", "cpu", "--max-length", "32", "--max-prompt-length", "16",
                       "--max-new-tokens", "4", "--gradient-accumulation-steps", "2",
                       "--lora-rank", "4", "--max-steps", "2", "--epochs", "2", *extra])


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_complete_classic_pipeline_and_dpo(tiny_model, tmp_path):
    sft = write_rows(tmp_path / "sft.jsonl", [{"prompt": "Hello", "response": "good"},
                                           {"prompt": "Test", "response": "yes"}])
    preference = write_rows(tmp_path / "preference.jsonl", [
        {"prompt": "Hello", "chosen": "good", "rejected": "bad"},
        {"prompt": "Test", "chosen": "yes", "rejected": "no"}])
    train(arguments("sft", tiny_model, sft, tmp_path / "sft", ["--eval-data", str(sft)]))
    train(arguments("reward", tiny_model, preference, tmp_path / "reward", ["--eval-data", str(preference)]))
    summary = train_ppo(arguments("ppo", tmp_path / "sft", sft, tmp_path / "ppo",
                                  ["--reward-model", str(tmp_path / "reward"), "--max-steps", "4"]))
    assert summary["optimizer_steps"] == 4
    metrics = [json.loads(line) for line in (tmp_path / "ppo/metrics.jsonl").read_text().splitlines()]
    assert metrics[0]["sampled_kl"] == 0
    assert abs(metrics[2]["sampled_kl"]) > 0  # The reference stays at SFT after policy updates.
    train(arguments("dpo", tmp_path / "sft", preference, tmp_path / "dpo"))
    actor_before = load_file(str(tmp_path / "sft/adapter_model.safetensors"))
    actor_after = load_file(str(tmp_path / "ppo/actor/adapter_model.safetensors"))
    assert any(not torch.equal(actor_before[key], actor_after[key]) for key in actor_before)
    critic_before = load_file(str(tmp_path / "reward/reward_head.safetensors"))["weight"]
    critic_after = load_file(str(tmp_path / "ppo/critic/reward_head.safetensors"))["weight"]
    assert not torch.equal(critic_before, critic_after)
    critic_adapter_before = load_file(str(tmp_path / "reward/adapter_model.safetensors"))
    critic_adapter_after = load_file(str(tmp_path / "ppo/critic/adapter_model.safetensors"))
    assert any(not torch.equal(critic_adapter_before[key], critic_adapter_after[key]) for key in critic_adapter_before)
    assert not any("reference" in key for key in actor_after)
    with pytest.raises(ValueError, match="nonempty"):
        train(arguments("sft", tiny_model, sft, tmp_path / "sft"))


def test_partial_gradient_accumulation_matches_full_batch(tiny_model, tmp_path):
    data = write_rows(tmp_path / "sft.jsonl", [{"prompt": "Hello", "response": "good"},
                                             {"prompt": "Test", "response": "yes good answer"}])
    # Two microbatches in a partially filled accumulation group of eight.
    train(arguments("sft", tiny_model, data, tmp_path / "micro",
                    ["--max-steps", "1", "--gradient-accumulation-steps", "8"]))
    train(arguments("sft", tiny_model, data, tmp_path / "full",
                    ["--max-steps", "1", "--batch-size", "2", "--gradient-accumulation-steps", "1"]))
    micro = load_file(str(tmp_path / "micro/adapter_model.safetensors"))
    full = load_file(str(tmp_path / "full/adapter_model.safetensors"))
    for key in micro:
        torch.testing.assert_close(micro[key], full[key], atol=1e-7, rtol=1e-4)


def test_config_cli_precedence_and_validation(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_steps": 9, "gradient_checkpointing": True}))
    args = parse_args(["sft", "--data", "x", "--output", "y", "--config", str(config),
                       "--max-steps", "1", "--no-gradient-checkpointing"])
    assert args.max_steps == 1
    assert args.gradient_checkpointing is False
    config.write_text('{"max_steps":"bad"}')
    with pytest.raises(SystemExit):
        parse_args(["sft", "--data", "x", "--output", "y", "--config", str(config)])
    with pytest.raises(SystemExit):
        parse_args(["ppo", "--data", "x", "--output", "y", "--reward-model", "z", "--batch-size", "2"])
