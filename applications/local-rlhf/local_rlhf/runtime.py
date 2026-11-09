"""Bounded single-process training, metrics and checkpoint bookkeeping."""

import json
import random
import time
from pathlib import Path

import psutil
import torch


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def memory_metrics(device):
    metrics = {"rss_gib": psutil.Process().memory_info().rss / 1024 ** 3}
    if device.type == "mps":
        metrics.update(mps_allocated_gib=torch.mps.current_allocated_memory() / 1024 ** 3,
                       mps_driver_gib=torch.mps.driver_allocated_memory() / 1024 ** 3)
    return metrics


def validate_output(output):
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output already exists and is nonempty: {output}. Choose a new --output.")


class Run:
    def __init__(self, args, device):
        validate_output(args.output)
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.start = time.monotonic()
        self.device = device
        self.steps = 0
        self.peak_samples = {}
        settings = {**vars(args), "resolved_device": str(device), "torch_version": torch.__version__}
        (self.output / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    def log(self, **metrics):
        metrics = {**metrics, "elapsed_seconds": round(time.monotonic() - self.start, 3),
                   **memory_metrics(self.device)}
        for key, value in metrics.items():
            if key.endswith("_gib"):
                self.peak_samples[key] = max(self.peak_samples.get(key, 0), value)
        with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(json.dumps(metrics, allow_nan=False), flush=True)

    def finish(self, **extra):
        summary = {"optimizer_steps": self.steps, "elapsed_seconds": time.monotonic() - self.start,
                   "device": str(self.device), "max_sampled_memory_gib": self.peak_samples, **extra}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def batches(rows, batch_size, seed, epoch):
    indices = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(indices)
    for offset in range(0, len(indices), batch_size):
        yield [rows[index] for index in indices[offset:offset + batch_size]]


def groups(iterator, size):
    group = []
    for item in iterator:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


def backward(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite loss; lower the learning rate or shorten sequences")
    loss.backward()


def optimizer_step(optimizer, parameters, max_norm):
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
