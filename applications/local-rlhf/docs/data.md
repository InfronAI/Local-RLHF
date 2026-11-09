# Local dataset formats

All inputs are UTF-8 JSONL: one JSON object per nonempty line. No dataset is downloaded automatically. The files in `data/` are small, synthetic demonstration examples.

## SFT

Use a prompt and response:

```json
{"prompt":"What is LoRA?","response":"LoRA fine-tunes a model using small low-rank weight updates."}
```

Or a chat conversation ending in a user/assistant pair:

```json
{"messages":[{"role":"system","content":"Answer concisely."},{"role":"user","content":"What is LoRA?"},{"role":"assistant","content":"Low-rank adaptation for efficient fine-tuning."}]}
```

Earlier turns are context. Only the final assistant response and EOS receive loss. Supported roles are `system`, `user`, and `assistant`; content must be a nonempty string.

## Reward Model and DPO

```json
{"prompt":"What is one plus two?","chosen":"Three.","rejected":"Four."}
```

`chosen` and `rejected` must be nonempty response strings for the same prompt. These commands expect plain response strings, not full chat transcripts in those fields. Pairs that become identical after tokenization/truncation are rejected rather than silently producing no preference signal.

## PPO

Only a prompt is required:

```json
{"prompt":"Explain why we keep a separate test set."}
```

Responses are sampled from the current actor, then scored by the fixed reward model. PPO uses the SFT actor tokenizer and requires the reward checkpoint to have the same vocabulary, EOS token and chat template.

## Lengths, padding and splits

Prompts are formatted using the model's chat template, then truncated from the left to `max_prompt_length`. For tokenizers without a template, a simple `User: ... Assistant:` format is used. Responses are truncated from the right to fit `max_length`, reserving one position for EOS. Very long prompts may lose earlier system instructions or dialogue turns; prepare appropriately sized data.

Training batches use right padding, explicit attention masks and `-100` padding labels. EOS remains a real training/scoring token even when it has the same ID as PAD. PPO generates one sequence at a time, so rollout tensors need no padding.

There is no implicit random train/test split. Supply separate files to `--data` and `--eval-data` for SFT, reward or DPO. Avoid overlapping examples when measuring generalization. `--max-samples` applies separately to each input file.

Malformed JSON includes its file and line number in the error. Missing fields, empty responses and invalid message roles fail before the model is trained.
