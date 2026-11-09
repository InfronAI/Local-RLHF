"""Local RLHF: the complete SFT, reward modeling and PPO workflow on a Mac."""

import os

# Set before torch is imported. Unsupported MPS ops may use CPU kernels.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.2.0"
