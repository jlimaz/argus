"""Convert training observations into the items Laya's fine-tuning script loads.

The Panoradio observations supply the eight signal types Laya is asked to tell
apart, and the sigid catalogue supplies the "other" class. A tenth of the data is
set aside as the test file. Every item is tokenised exactly as the pipeline
tokenises an observation at inference time.
"""

import argparse
import collections
import json
import random
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from laya.common import QTYPES, build_sequence
from transformers import AutoTokenizer

from finetune.sources.common import ROOT, TRAIN, read_jsonl, write_jsonl
from pipeline.laya import HEAD_MAX_LEN, MAX_LEN, describe

# Fine-tuning always starts from the published checkpoint.
BASE_CHECKPOINT = "convaiinnovations/laya"
QUESTIONS = ROOT / "data" / "questions_20.json"
OUT = ROOT / "finetune" / "out"
TEST = ROOT / "data" / "test_observations.jsonl"
PANORADIO = TRAIN / "panoradio_hf.jsonl"
SIGID = TRAIN / "sigid.jsonl"
QUESTION = "mode"
SEED = 20
TEST_SHARE = 0.1
# With every sigid row, "other" would be two thirds of the data.
OTHER_ROWS = 1500

# Panoradio signal name: the answer option it belongs to. Modes that centre
# frequency, bandwidth and modulation cannot tell apart share an option.
MODE_BY_SIGNAL = {
    "Morse Code (CW)": "morse",
    "PSK31": "psk",
    "PSK63": "psk",
    "QPSK31": "psk",
    "RTTY 45 baud 170 Hz": "rtty",
    "Olivia 8/250": "olivia",
    "Olivia 16/500": "olivia",
    "Olivia 16/1000": "olivia",
    "Olivia 32/1000": "olivia",
    "DominoEX 11": "dominoex",
    "MT63-1000": "mt63",
    "NAVTEX (SITOR-B)": "navtex",
    "HF weather fax": "weather_fax",
}
# Catalogue entries for the same modes under a generic name. Calling them "other"
# would contradict the Panoradio rows, so they are left out.
EXCLUDED_SIGID = {
    "Olivia",
    "MT63",
    "DominoEX",
    "DominoF",
    "Radio Teletype (RTTY)",
    "RTTYM",
    "Coherent CW",
    "Phase Shift Keying (PSK)",
    "Coherent BPSK",
    "PSK-AM",
    "SITOR-B",
}


def spread(rows: list[dict], rng: random.Random) -> list[dict]:
    """Order rows so any prefix covers as many signals as possible."""
    by_signal = collections.defaultdict(list)
    for row in rows:
        by_signal[row["label_signal"]].append(row)
    queues = list(by_signal.values())
    for queue in queues:
        rng.shuffle(queue)
    rng.shuffle(queues)
    ordered = []
    while queues:
        ordered.extend(queue.pop() for queue in queues)
        queues = [queue for queue in queues if queue]
    return ordered


def split_rows(panoradio: list[dict], sigid: list[dict]) -> tuple[list[dict], list[dict]]:
    """Label every row with its answer option and split them into training and test rows."""
    rng = random.Random(SEED)
    by_mode = collections.defaultdict(list)
    for row in panoradio:
        mode = MODE_BY_SIGNAL[row["label_signal"]]
        by_mode[mode].append({**row, "label_mode": mode})
    train, test = [], []
    for mode in sorted(by_mode):
        rows = by_mode[mode]
        rng.shuffle(rows)
        kept = round(TEST_SHARE * len(rows))
        test.extend(rows[:kept])
        train.extend(rows[kept:])

    # "other" is tested on signal types the model never trained on.
    other = [{**row, "label_mode": "other"} for row in sigid if row["label_signal"] not in EXCLUDED_SIGID]
    names = sorted({row["label_signal"] for row in other})
    rng.shuffle(names)
    held_out = set(names[: round(TEST_SHARE * len(names))])
    test_quota = round(TEST_SHARE * OTHER_ROWS)
    test.extend(spread([row for row in other if row["label_signal"] in held_out], rng)[:test_quota])
    train.extend(spread([row for row in other if row["label_signal"] not in held_out], rng)[: OTHER_ROWS - test_quota])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def build_item(tokenizer, row: dict, question: dict, rng: random.Random) -> dict:
    """One training item, with the options in a random order."""
    labels = list(question["criteria"])
    rng.shuffle(labels)
    internal = {
        "t": question["type"],
        "ins": question["instructions"],
        "crit": {label: question["criteria"][label] for label in labels},
    }
    ids, markers = build_sequence(tokenizer, describe(row), internal, MAX_LEN, HEAD_MAX_LEN)
    if len(markers) != len(labels):
        raise SystemExit("An item lost options to truncation; check MAX_LEN and HEAD_MAX_LEN.")
    label = labels.index(row["label_mode"])
    return {
        "ids": ids,
        "markers": markers,
        "qtype": QTYPES[question["type"]],
        "target": [float(index == label) for index in range(len(labels))],
        "label": label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--test", type=Path, default=TEST, help="where to write the test rows")
    args = parser.parse_args()

    question = json.loads(args.questions.read_text())[QUESTION]
    train, test = split_rows(read_jsonl(PANORADIO), read_jsonl(SIGID))
    missing = {row["label_mode"] for row in train + test} ^ set(question["criteria"])
    if missing:
        raise SystemExit(f"The question's options and the data's labels differ: {sorted(missing)}")
    for name, rows in (("training", train), ("test", test)):
        counts = collections.Counter(row["label_mode"] for row in rows)
        print(f"{name}: {len(rows)} rows, {dict(sorted(counts.items()))}")

    tokenizer = AutoTokenizer.from_pretrained(Path(snapshot_download(BASE_CHECKPOINT)) / "tokenizer")
    rng = random.Random(SEED)
    items = [build_item(tokenizer, row, question, rng) for row in train]
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(items, args.out / "train_items.pt")
    write_jsonl(args.out / "train_rows.jsonl", train)
    write_jsonl(args.test, test)
    print(f"wrote {len(items)} items to {args.out / 'train_items.pt'}")
    print(f"test set: {args.test}")


if __name__ == "__main__":
    main()
