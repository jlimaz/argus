"""Measure a Laya checkpoint on the test observations, optionally against a baseline."""

import argparse
import collections
import json
from pathlib import Path

import laya

from finetune.sources.common import ROOT, read_jsonl
from pipeline.laya import classify, pick_device, release

QUESTIONS = ROOT / "data" / "questions_20.json"
TEST = ROOT / "data" / "test_observations.jsonl"
BASE_CHECKPOINT = "convaiinnovations/laya"
QUESTION = "mode"
# A fine-tune is good enough when it clears both bars on the test set.
MIN_ACCURACY = 0.50
MAX_SHARE_OF_ONE_MODE = 0.50


def measure(checkpoint: str, rows: list[dict], questions: dict) -> dict:
    """Classify every row with one checkpoint and summarise the outcome."""
    agent = laya.load(checkpoint, device=pick_device())
    print(f"running {checkpoint} on {agent.device}")
    predicted = [answer[QUESTION][0] for answer in classify(agent, rows, questions)]
    release(agent)
    truth = [row["label_mode"] for row in rows]
    per_mode = {}
    for mode in questions[QUESTION]["criteria"]:
        total = truth.count(mode)
        hits = sum(p == t == mode for p, t in zip(predicted, truth))
        per_mode[mode] = (hits, total, predicted.count(mode))
    top_mode, top_count = collections.Counter(predicted).most_common(1)[0]
    # The options differ in size, so also average the accuracy of each one.
    shares = [hits / total for hits, total, _ in per_mode.values() if total]
    return {
        "accuracy": sum(p == t for p, t in zip(predicted, truth)) / len(rows),
        "balanced_accuracy": sum(shares) / len(shares),
        "per_mode": per_mode,
        "top_mode": top_mode,
        "top_share": top_count / len(rows),
    }


def report(name: str, result: dict) -> None:
    print(f"\n{name}")
    print(f"  overall accuracy: {result['accuracy']:.1%}")
    print(f"  balanced accuracy: {result['balanced_accuracy']:.1%}")
    print(f"  most predicted:   {result['top_mode']} ({result['top_share']:.1%} of answers)")
    print(f"  {'signal type':<20} {'found':>11} {'predicted':>10}")
    for mode, (hits, total, predicted) in result["per_mode"].items():
        print(f"  {mode:<20} {hits:>5}/{total:<5} {predicted:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    parser.add_argument("--baseline", help="a second checkpoint to compare against")
    parser.add_argument("--test", type=Path, default=TEST)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--limit", type=int, help="use only the first N test rows")
    args = parser.parse_args()

    rows = read_jsonl(args.test)[: args.limit]
    questions = json.loads(args.questions.read_text())
    result = measure(args.checkpoint, rows, questions)
    report(f"{args.checkpoint} on {len(rows)} test rows", result)
    if args.baseline:
        baseline = measure(args.baseline, rows, questions)
        report(f"baseline {args.baseline}", baseline)
        print(f"\nchange in overall accuracy: {result['accuracy'] - baseline['accuracy']:+.1%}")
    passed = result["accuracy"] >= MIN_ACCURACY and result["top_share"] <= MAX_SHARE_OF_ONE_MODE
    print(
        f"\nbar (accuracy >= {MIN_ACCURACY:.0%}, no signal type above "
        f"{MAX_SHARE_OF_ONE_MODE:.0%} of answers): {'PASSED' if passed else 'not met'}"
    )


if __name__ == "__main__":
    main()
