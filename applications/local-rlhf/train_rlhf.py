"""Run the classic SFT -> Reward Model -> PPO pipeline in separate processes."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(root / "configs/m1_16gb.json"))
    parser.add_argument("--model", help="Override base model for both SFT and reward")
    parser.add_argument("--sft-data", default=str(root / "data/sft.jsonl"))
    parser.add_argument("--preference-data", default=str(root / "data/preferences.jsonl"))
    parser.add_argument("--prompt-data", default=str(root / "data/prompts.jsonl"))
    parser.add_argument("--output", default=str(root / "output/local-rlhf"))
    parser.add_argument("--device", choices=["auto", "mps", "cpu"])
    parser.add_argument("--max-steps", type=int, help="Optimizer update cap per stage")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be new/empty: {output}")
    for path in [args.config, args.sft_data, args.preference_data, args.prompt_data]:
        if not Path(path).is_file():
            parser.error(f"Missing input: {path}")
    common = ["--config", str(Path(args.config).resolve())]
    for flag in ["device", "max_steps", "epochs"]:
        if getattr(args, flag) is not None:
            common += ["--" + flag.replace("_", "-"), str(getattr(args, flag))]
    base = ["--model", args.model] if args.model else []
    stages = [
        ["sft", *common, *base, "--data", str(Path(args.sft_data).resolve()), "--output", str(output / "sft")],
        ["reward", *common, *base, "--data", str(Path(args.preference_data).resolve()), "--output", str(output / "reward")],
        ["ppo", *common, "--model", str(output / "sft"), "--reward-model", str(output / "reward"),
         "--data", str(Path(args.prompt_data).resolve()), "--output", str(output / "ppo")],
    ]
    for stage in stages:
        print(f"\nRunning {stage[0]}...", flush=True)
        subprocess.run([sys.executable, "-m", "local_rlhf", *stage], cwd=root, check=True)
    print(f"\nCompleted SFT -> Reward Model -> PPO. Actor: {output / 'ppo/actor'}")


if __name__ == "__main__":
    main()
