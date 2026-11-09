"""Compare real-model CPU/MPS DPO gradients across different sequence lengths.

Unlike a tiny-model smoke test, this checks the large-vocabulary backward path.
Weights must be downloaded once; use HF_HUB_OFFLINE=1 to reuse the local cache.
"""

import argparse
import gc
import json
from pathlib import Path

import local_rlhf  # Configure the backend environment before importing torch.
import torch

from local_rlhf.data import prepare_rows, read_jsonl
from local_rlhf.models import load_policy, load_tokenizer, snapshot_reference
from local_rlhf.runtime import seed_everything, trainable_parameters
from local_rlhf.trainers import compute_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/preferences.jsonl"))
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--output", default="output/mps-gradients.json")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("This check requires a Mac with MPS")
    if args.max_samples <= 0:
        parser.error("max-samples must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new output file")
    tokenizer = load_tokenizer(args.model)
    rows = prepare_rows(read_jsonl(args.data, args.max_samples), tokenizer, "dpo", 256, 128)
    reference, records = [], []
    for name in ["cpu", "mps"]:
        seed_everything(42)
        device = torch.device(name)
        model = load_policy(args.model, device)
        snapshot_reference(model)
        model.train()
        for index, row in enumerate(rows):
            model.zero_grad(set_to_none=True)
            loss, _ = compute_loss("dpo", model, [row], tokenizer, device, 0.1)
            loss.backward()
            gradient = torch.cat([p.grad.detach().cpu().flatten() for p in trainable_parameters(model)
                                  if p.grad is not None]).double()
            if name == "cpu":
                reference.append(gradient)
            else:
                ratio = float(gradient.norm() / reference[index].norm())
                cosine = float(torch.nn.functional.cosine_similarity(reference[index], gradient, dim=0))
                record = {"row": index + 1, "cpu_norm": float(reference[index].norm()),
                          "mps_norm": float(gradient.norm()), "norm_ratio": ratio, "cosine": cosine,
                          "passed": abs(ratio - 1.0) < 0.02 and cosine > 0.999}
                records.append(record)
                print(json.dumps(record), flush=True)
        del model, loss
        gc.collect()
        if name == "mps":
            torch.mps.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"torch": torch.__version__, "model": args.model, "comparisons": records,
              "passed": all(record["passed"] for record in records)}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit("CPU/MPS gradient mismatch; use the tested dependency versions or --device cpu")
    print(f"PASS: {len(records)} CPU/MPS gradient comparisons. Results: {output}")


if __name__ == "__main__":
    main()
