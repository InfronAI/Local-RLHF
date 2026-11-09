"""Offline SFT -> RM -> PPO, DPO and saved-actor inference on a tiny random model."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from local_rlhf.testing import make_tiny_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--output", default="output/smoke")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new/empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    base = make_tiny_model(output / "base")
    config = {"model": base, "device": args.device, "max_length": 32, "max_prompt_length": 16,
              "max_new_tokens": 4, "batch_size": 1, "gradient_accumulation_steps": 2,
              "max_steps": 2, "epochs": 2, "lora_rank": 4}
    (output / "config.json").write_text(json.dumps(config))
    records = {
        "sft": [{"prompt": "What is one plus two ?", "response": "three"},
                {"prompt": "Hello", "response": "Hello world"}],
        "preferences": [{"prompt": "What is one plus two ?", "chosen": "three", "rejected": "four"},
                        {"prompt": "Hello", "chosen": "good", "rejected": "bad"}],
        "prompts": [{"prompt": "What is one plus two ?"}, {"prompt": "Hello"}],
    }
    for name, rows in records.items():
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "train_rlhf.py"), "--config", str(output / "config.json"),
               "--sft-data", str(output / "sft.jsonl"), "--preference-data", str(output / "preferences.jsonl"),
               "--prompt-data", str(output / "prompts.jsonl"), "--output", str(output / "pipeline")]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, "-m", "local_rlhf", "dpo", "--config", str(output / "config.json"),
                    "--model", str(output / "pipeline/sft"), "--data", str(output / "preferences.jsonl"),
                    "--output", str(output / "dpo")], check=True)
    subprocess.run([sys.executable, "-m", "local_rlhf", "chat", "--model", str(output / "pipeline/ppo/actor"),
                    "--device", args.device, "--prompt", "Hello", "--max-prompt-length", "16",
                    "--max-new-tokens", "4"], check=True)
    print(f"PASS: offline training and checkpoint inference completed. Artifacts: {output}")


if __name__ == "__main__":
    main()
