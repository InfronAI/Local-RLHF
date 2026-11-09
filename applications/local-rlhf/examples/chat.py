"""Chat with a trained SFT, DPO or PPO actor checkpoint.

From applications/local-rlhf after installation:
    python examples/chat.py --model output/local-rlhf/ppo/actor --device mps
    python examples/chat.py --model output/sft --prompt "What is LoRA?"

Omit --prompt for multi-turn interactive chat. Type exit/quit to finish.
The frozen base model must be available locally or in the Hugging Face cache.
"""

import sys

from local_rlhf.cli import main


if __name__ == "__main__":
    main(["chat", *sys.argv[1:]])
