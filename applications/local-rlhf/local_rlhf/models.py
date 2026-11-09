"""Frozen transformer backbones with independently trainable LoRA roles."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_device(requested):
    if requested == "auto":
        requested = "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable: use native arm64 Python + macOS, or --device cpu")
    return torch.device(requested)


def load_tokenizer(source, revision=None):
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_policy(source, device, rank=8, checkpointing=True, trainable=True, revision=None):
    source = str(Path(source).resolve()) if Path(source).exists() else source
    is_adapter = (Path(source) / "adapter_config.json").is_file()
    config = PeftConfig.from_pretrained(source) if is_adapter else None
    base_source = config.base_model_name_or_path if config else source
    base = AutoModelForCausalLM.from_pretrained(
        base_source, revision=config.revision if config else revision,
        dtype=torch.float32, attn_implementation="eager", trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    if config:
        model = PeftModel.from_pretrained(base, source, is_trainable=trainable)
    else:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2,
            lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none",
            revision=getattr(base.config, "_commit_hash", None) or revision,
        )
        model = get_peft_model(base, config)
    # Dropout must not perturb PPO old/new policy ratios or DPO reference scores.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.config.use_cache = False
    if trainable and checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if not trainable:
        model.requires_grad_(False)
        model.eval()
    return model.to(device)


def snapshot_reference(model):
    """Snapshot SFT/RM LoRA weights, sharing only the immutable base parameters."""
    weights = {key: value.detach().clone() for key, value in
               get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False).items()}
    model.add_adapter("reference", deepcopy(model.peft_config["default"]))
    set_peft_model_state_dict(model, weights, adapter_name="reference")
    model.set_adapter("default")
    for name, parameter in model.named_parameters():
        if ".reference." in name:
            parameter.requires_grad_(False)


@contextmanager
def reference_policy(model):
    was_training = model.training
    model.set_adapter("reference")
    model.eval()
    for name, parameter in model.named_parameters():
        if ".reference." in name:
            parameter.requires_grad_(False)
    try:
        with torch.no_grad():
            yield model
    finally:
        model.set_adapter("default")
        model.train(was_training)


class RewardModel(torch.nn.Module):
    """Token value head; reward is the score at the final non-padding token."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.head = torch.nn.Linear(policy.config.hidden_size, 1, bias=False)
        torch.nn.init.normal_(self.head.weight, std=0.02)
        self.head.to(next(policy.parameters()).device)

    def values(self, input_ids, attention_mask, head=None):
        # Access injected transformer layers directly, avoiding the vocabulary logits.
        backbone = self.policy.get_base_model().base_model
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False, return_dict=True).last_hidden_state
        return (self.head if head is None else head)(hidden).squeeze(-1)

    def forward(self, input_ids, attention_mask, head=None):
        values = self.values(input_ids, attention_mask, head=head)
        # Supports both left/right padding; EOS may equal PAD without losing EOS.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        last = (positions * attention_mask).max(dim=1).values
        return values.gather(1, last.unsqueeze(1)).squeeze(1)


def load_reward(source, device, rank=8, checkpointing=True, require_head=False, revision=None):
    model = RewardModel(load_policy(source, device, rank, checkpointing, revision=revision))
    head_path = Path(source) / "reward_head.safetensors"
    if head_path.is_file():
        model.head.load_state_dict(load_file(str(head_path)))
    elif require_head:
        raise ValueError(f"{source}: missing trained reward_head.safetensors; run reward first")
    return model


def save_policy(model, tokenizer, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Embeddings are frozen and vocabulary is unchanged; avoid Hub probing when offline.
    model.save_pretrained(output, selected_adapters=["default"], safe_serialization=True,
                          save_embedding_layers=False)
    tokenizer.save_pretrained(output)


def save_reward(model, tokenizer, output):
    save_policy(model.policy, tokenizer, output)
    save_file({key: value.detach().cpu().contiguous() for key, value in model.head.state_dict().items()},
              str(Path(output) / "reward_head.safetensors"))
