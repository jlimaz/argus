"""Training: `python -m mcls.train --config configs/X.toml` (PLAN.md 5.3, 8).

Importing this package does not import torch. The torch modules (loop, losses,
optimizers, ema, checkpoint, adabn) are imported only when a neural run starts.
"""
