"""Export a trained neural run's best weights to ONNX, and check it against torch.

    python scripts/export_onnx.py runs/<name> [--manifest data/manifest_w5.jsonl] [--check 64]

Writes runs/<name>/model.onnx: input "iq", float32 (batch, 2, 4096) (I then Q, a
unit-RMS window as the front end cuts it); output "logits", float32 (batch,
classes), before temperature. The spectrogram of the 2D CNN is part of the graph,
so the caller only feeds I/Q. Also writes runs/<name>/model.onnx.json with the
class order and the check result.

The check runs onnxruntime and torch on --check windows (real test windows read
from the manifest's shards when --manifest is given, else random noise) and fails
when any logit differs by more than 1e-3. Needs torch, onnx and onnxruntime:
run it on CRC, not on the laptop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def test_windows(manifest: Path, count: int, seed: int = 0) -> np.ndarray:
    rows = []
    with open(manifest) as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "test":
                rows.append(row)
    rng = np.random.default_rng(seed)
    picked = [rows[i] for i in rng.choice(len(rows), size=min(count, len(rows)), replace=False)]
    out = []
    for row in picked:
        shard = np.load(manifest.parent / row["shard"], mmap_mode="r")
        window = np.asarray(shard[row["offset"] : row["offset"] + row["length"]])  # shards are flat sample arrays
        out.append(np.stack([window.real, window.imag]).astype(np.float32))
    return np.stack(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--check", type=int, default=64)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    import onnxruntime as ort
    import torch

    from mcls.models.heads import main_logits
    from mcls.train import checkpoint as ckpt

    payload = ckpt.load_best(args.run / ckpt.BEST)
    model = ckpt.build_from(payload)

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, iq):
            return main_logits(self.inner(iq))

    wrapped = LogitsOnly(model).eval()
    out_path = args.run / "model.onnx"
    example = torch.zeros(2, 2, 4096)
    torch.onnx.export(wrapped, (example,), str(out_path), input_names=["iq"], output_names=["logits"],
                      dynamic_axes={"iq": {0: "batch"}, "logits": {0: "batch"}}, opset_version=args.opset,
                      dynamo=False)

    if args.manifest:
        x = test_windows(args.manifest, args.check)
        source = f"{len(x)} real test windows from {args.manifest.name}"
    else:
        x = np.random.default_rng(0).standard_normal((args.check, 2, 4096)).astype(np.float32)
        source = f"{len(x)} random windows"
    with torch.no_grad():
        expected = wrapped(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    got = session.run(["logits"], {"iq": x})[0]
    diff = float(np.max(np.abs(got - expected)))
    agree = float(np.mean(got.argmax(1) == expected.argmax(1)))
    info = {"classes": list(payload["classes"]), "model": payload["model"], "epoch": payload.get("epoch"),
            "val_macro_f1": payload.get("val_macro_f1"), "check": source, "max_abs_logit_diff": diff,
            "argmax_agreement": agree, "opset": args.opset}
    (args.run / "model.onnx.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))
    if diff > 1e-3:
        print(f"FAILED: ONNX and torch differ by {diff:.2e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
