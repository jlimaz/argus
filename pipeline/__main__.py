import json
import sys
import time
from pathlib import Path

from pipeline.inputs import MAX_OBSERVATIONS
from pipeline.laya import classify, load_agent, release
from pipeline.ollama import DEFAULT_QUESTION, judge, unload, warmup, warmup_seconds

ROOT = Path(__file__).resolve().parent.parent


def main():
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    # Laya and the Ollama model do not both fit on a small GPU, so they take turns.
    started = time.perf_counter()
    unload()
    timings = [("ollama unload", time.perf_counter() - started)]
    lines = (ROOT / "data" / "test_observations.jsonl").read_text().splitlines()
    observations = [json.loads(line) for line in lines if line.strip()][:MAX_OBSERVATIONS]
    questions = json.loads((ROOT / "data" / "questions_20.json").read_text())
    started = time.perf_counter()
    agent = load_agent()
    timings.append(("laya load", time.perf_counter() - started))
    started = time.perf_counter()
    answers = classify(agent, observations, questions)
    timings.append((f"laya classify [{len(observations)}]", time.perf_counter() - started))
    release(agent)
    warmup()
    timings.append(("ollama warmup", warmup_seconds()))
    classified = []
    correct = 0
    for observation, answer in zip(observations, answers):
        mode, confidence = answer["mode"]
        if mode == observation["label_mode"]:
            correct += 1
        classified.append(
            {
                "id": observation["id"],
                "center_frequency_hz": observation["center_frequency_hz"],
                "bandwidth_hz": observation["bandwidth_hz"],
                "modulation": observation["modulation"],
                "mode": mode,
                "confidence": confidence,
            }
        )
    started = time.perf_counter()
    judgment = judge(classified, list(questions["mode"]["criteria"]), question)
    timings.append(("ollama judge", time.perf_counter() - started))
    morse = sum(observation["label_mode"] == "morse" for observation in observations)
    for item in classified:
        print(json.dumps(item))
    print(f"\033[32m{json.dumps({'judgment': judgment})}\033[0m")
    print(
        f"laya {correct}/{len(observations)} "
        f"expected morse: {morse} "
        f"expected not morse: {len(observations) - morse}"
    )
    for step, seconds in timings:
        print(f"{step}: {seconds:.2f}s", file=sys.stderr)
    print(f"total: {sum(seconds for _, seconds in timings):.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
