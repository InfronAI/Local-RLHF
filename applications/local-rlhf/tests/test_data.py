import pytest
import torch

from local_rlhf.data import collate, prepare_rows, read_jsonl
from local_rlhf.models import load_tokenizer


def test_prompt_mask_and_eos_when_pad_equals_eos(tiny_model):
    tokenizer = load_tokenizer(tiny_model)
    tokenizer.pad_token = tokenizer.eos_token
    rows = prepare_rows([{"prompt": "Hello", "response": "good"},
                         {"prompt": "What is one plus two ?", "response": "three"}],
                        tokenizer, "sft", 32, 16)
    batch = collate(rows, tokenizer.pad_token_id, torch.device("cpu"))
    for i, row in enumerate(rows):
        end = len(row["input_ids"]) - 1
        assert batch["labels"][i, end].item() == tokenizer.eos_token_id
        assert batch["attention_mask"][i, end].item() == 1
        assert batch["labels"][i, 0].item() == -100
        assert batch["attention_mask"][i, end + 1:].sum().item() == 0
        assert torch.all(batch["labels"][i, end + 1:] == -100)


def test_multiturn_masks_earlier_assistant(tiny_model):
    tokenizer = load_tokenizer(tiny_model)
    row = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "good"},
                         {"role": "user", "content": "What is one plus two ?"},
                         {"role": "assistant", "content": "three"}]}
    encoded = prepare_rows([row], tokenizer, "sft", 32, 24)[0]
    labels = [label for label in encoded["labels"] if label != -100]
    assert labels == tokenizer.encode("three", add_special_tokens=False) + [tokenizer.eos_token_id]


def test_bad_preferences_and_bounded_read(tiny_model, tmp_path):
    tokenizer = load_tokenizer(tiny_model)
    with pytest.raises(ValueError, match="identical"):
        prepare_rows([{"prompt": "Hello", "chosen": "good", "rejected": "good"}], tokenizer, "reward", 32, 16)
    path = tmp_path / "rows.jsonl"
    path.write_text('{"prompt":"Hello"}\ninvalid json\n')
    assert len(read_jsonl(path, 1)) == 1
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(path)
