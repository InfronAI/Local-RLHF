"""Local JSONL input and response-only token masks."""

import json
from pathlib import Path

import torch


def read_jsonl(path, limit=0):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"{path}: dataset is empty")
    return rows


def require_text(row, key):
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Each row needs a nonempty string field {key!r}")
    return value


def prompt_messages(row):
    if "messages" in row:
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("messages needs a conversation ending in an assistant response")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Invalid message role")
            require_text(message, "content")
        if messages[-1]["role"] != "assistant" or messages[-2]["role"] != "user":
            raise ValueError("messages must end with user, assistant")
        return messages[:-1], messages[-1]["content"]
    return [{"role": "user", "content": require_text(row, "prompt")}], row.get("response")


def prompt_ids(tokenizer, messages, max_length):
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    else:
        text = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages)
        ids = tokenizer.encode(text + "Assistant:", add_special_tokens=True)
    if not ids:
        raise ValueError("Prompt produced no tokens")
    return ids[-max_length:]


def encode_response(tokenizer, prompt, response, max_length):
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response/chosen/rejected must be a nonempty string")
    available = max_length - len(prompt)
    if available < 2:
        raise ValueError("max_length must leave room for response tokens and EOS")
    answer = tokenizer.encode(response, add_special_tokens=False)[:available - 1]
    if not answer:
        raise ValueError("Response produced no tokens")
    answer.append(tokenizer.eos_token_id)
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


def prepare_rows(rows, tokenizer, stage, max_length, max_prompt_length):
    result = []
    for index, row in enumerate(rows, 1):
        try:
            messages, answer = prompt_messages(row)
            prompt = prompt_ids(tokenizer, messages, max_prompt_length)
            if stage == "sft":
                result.append(encode_response(tokenizer, prompt, answer, max_length))
            elif stage in {"reward", "dpo"}:
                chosen = encode_response(tokenizer, prompt, require_text(row, "chosen"), max_length)
                rejected = encode_response(tokenizer, prompt, require_text(row, "rejected"), max_length)
                if chosen["input_ids"] == rejected["input_ids"]:
                    raise ValueError("chosen and rejected are identical after tokenization/truncation")
                result.append({"chosen": chosen, "rejected": rejected})
            else:
                result.append({"input_ids": prompt})
        except ValueError as exc:
            raise ValueError(f"Dataset row {index}: {exc}") from exc
    return result


def collate(rows, pad_id, device):
    """Right padding; attention masks never infer padding from EOS IDs."""
    width = max(len(row["input_ids"]) for row in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        length = len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * (width - length))
        masks.append([1] * length + [0] * (width - length))
        labels.append(row.get("labels", [-100] * length) + [-100] * (width - length))
    return {key: torch.tensor(value, dtype=torch.long, device=device)
            for key, value in zip(("input_ids", "attention_mask", "labels"), (ids, masks, labels))}
