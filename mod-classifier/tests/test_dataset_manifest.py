import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mcls import manifest
from mcls.contracts import WINDOW_SAMPLES, ManifestRow
from mcls.dataset import (
    WindowStore,
    audit_window_hashes,
    build,
    build_with_report,
    cap_per_class,
    collect_rows,
    holdout_emitters,
    label_indices,
    row_weights,
)
from mcls.dataset import build_manifest
from mcls.dataset.fakes import make_fake_shards


@pytest.fixture
def shards(tmp_path):
    make_fake_shards(tmp_path / "a", seed=1, dataset="ds_a", recordings_per_class=6)
    make_fake_shards(tmp_path / "b", seed=2, dataset="ds_b", recordings_per_class=3, emitters=())
    return [tmp_path / "a", tmp_path / "b"]


def test_collect_resolves_relative_shards_and_sorts(shards):
    rows = collect_rows(shards)
    assert len(rows) == (6 + 3) * 2 * 6
    assert [r.window_id for r in rows] == sorted(r.window_id for r in rows)
    assert all(Path(r.shard).is_absolute() and Path(r.shard).exists() for r in rows)


def test_collect_reads_front_end_rows_without_split(tmp_path):
    # e4.md: rows files are <stem>.sNNN.rows.jsonl with every ManifestRow field except "split".
    make_fake_shards(tmp_path, dataset="ds")
    rows_file = sorted(tmp_path.glob("*.s000.rows.jsonl"))[0]
    assert "split" not in json.loads(rows_file.read_text().splitlines()[0])
    # Other .jsonl files (a manifest written into the same folder) are not front-end rows.
    (tmp_path / "manifest.jsonl").write_text('{"not": "a row"}\n')
    # Hidden folders (partial outputs of killed CRC tasks) are skipped.
    make_fake_shards(tmp_path / ".key.partial.123", dataset="partial")
    rows = collect_rows([tmp_path])
    assert len(rows) == 2 * 4 * 6 and {r.split for r in rows} == {"train"}


def test_collect_rejects_duplicate_window_ids(tmp_path):
    make_fake_shards(tmp_path / "a", dataset="same")
    make_fake_shards(tmp_path / "b", dataset="same")
    with pytest.raises(ValueError, match="appears in"):
        collect_rows([tmp_path / "a", tmp_path / "b"])


def test_cap_per_class_per_dataset_is_deterministic_and_order_free(shards):
    rows = collect_rows(shards)
    capped = cap_per_class(rows, 10, seed=3)
    counts = Counter((r.dataset, r.label) for r in capped)
    assert counts[("ds_a", "drone_link")] == 10 and counts[("ds_b", "wifi")] == 10
    again = cap_per_class(list(reversed(rows)), 10, seed=3)
    assert [r.window_id for r in again] == [r.window_id for r in capped]
    other = cap_per_class(rows, 10, seed=4)
    assert [r.window_id for r in other] != [r.window_id for r in capped]
    assert len(cap_per_class(rows, 0, seed=3)) == len(rows)


def test_build_splits_by_recording_and_cross_test(shards):
    rows = build(shards, seed=5, cross_test=["ds_b"], cap=0)
    manifest.check_leakage(rows)
    assert {r.split for r in rows if r.dataset == "ds_b"} == {"cross_test"}
    by_recording = {}
    for r in rows:
        by_recording.setdefault(r.recording_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_recording.values())


def test_holdout_emitter_goes_to_generalization_by_default(shards):
    rows = build(shards, seed=5, cap=0, holdout=["model_a"])
    held = [r for r in rows if r.emitter_id == "model_a"]
    assert held and {r.split for r in held} == {"generalization"}
    assert all(r.split != "generalization" for r in rows if r.emitter_id != "model_a")
    # The remaining drone_link recordings of ds_a (3, all model_b) still fill train, val and test.
    rest = {r.split for r in rows if r.dataset == "ds_a" and r.label == "drone_link" and r.emitter_id != "model_a"}
    assert rest == {"train", "val", "test"}
    manifest.check_leakage(rows)


def test_holdout_split_can_be_test_but_not_train(shards):
    rows = build(shards, seed=5, cap=0, holdout=["model_a"], holdout_split="test")
    assert {r.split for r in rows if r.emitter_id == "model_a"} == {"test"}
    with pytest.raises(ValueError, match="cannot go to"):
        build(shards, seed=5, cap=0, holdout=["model_a"], holdout_split="train")


@pytest.mark.parametrize("seed", range(8))
def test_stratified_every_class_with_three_recordings_in_each_split(tmp_path, seed):
    make_fake_shards(tmp_path / "a", dataset="ds_a", recordings_per_class=3, windows_per_recording=2)
    make_fake_shards(tmp_path / "b", dataset="ds_b", recordings_per_class=11, windows_per_recording=2)
    rows, report = build_with_report([tmp_path / "a", tmp_path / "b"], seed=seed, cap=0)
    assert report.warnings == []
    manifest.check_leakage(rows)
    for dataset, n in (("ds_a", 3), ("ds_b", 11)):
        for label in ("drone_link", "wifi"):
            per_split = Counter()
            for recording, split in {(r.recording_id, r.split) for r in rows if r.dataset == dataset and r.label == label}:
                per_split[split] += 1
            assert set(per_split) == {"train", "val", "test"}, (dataset, label, per_split)
            assert sum(per_split.values()) == n
            if n == 11:
                assert per_split["val"] == per_split["test"] == 2 and per_split["train"] == 7


def test_stratified_split_is_seeded_and_order_free(tmp_path):
    make_fake_shards(tmp_path, recordings_per_class=10, windows_per_recording=2)
    rows = collect_rows([tmp_path])
    reversed_rows = [ManifestRow.from_dict(r.to_dict()) for r in reversed(rows)]
    manifest.assign_splits_stratified(rows, 3)
    manifest.assign_splits_stratified(reversed_rows, 3)
    assert {r.window_id: r.split for r in rows} == {r.window_id: r.split for r in reversed_rows}
    manifest.assign_splits_stratified(reversed_rows, 4)
    assert {r.window_id: r.split for r in rows} != {r.window_id: r.split for r in reversed_rows}


def test_stratified_warns_for_small_classes(tmp_path):
    make_fake_shards(tmp_path, dataset="tiny", recordings_per_class=2, windows_per_recording=2)
    rows, report = build_with_report([tmp_path], seed=1, cap=0)
    assert len(report.warnings) == 2
    assert any("'drone_link'" in w and "tiny" in w for w in report.warnings)
    assert any("'wifi'" in w for w in report.warnings)
    manifest.check_leakage(rows)


def test_stratified_keeps_multi_label_recordings_whole(tmp_path):
    # A recording carrying two labels (e.g. GATEMAN: a GNSS file and a jammer file share a scenario id).
    rows = collect_rows([make_shards_dir(tmp_path)])
    for r in rows:
        if r.recording_id.endswith("_0") and r.label == "wifi":
            r.recording_id = r.recording_id.replace("wifi", "drone_link")
    manifest.assign_splits_stratified(rows, 2)
    manifest.check_leakage(rows)


def make_shards_dir(tmp_path):
    make_fake_shards(tmp_path, recordings_per_class=5, windows_per_recording=2)
    return tmp_path


def test_cross_test_and_generalization_are_left_alone_by_stratification(shards):
    rows = build(shards, seed=1, cap=0, cross_test=["ds_b"], holdout=["model_b"])
    assert {r.split for r in rows if r.dataset == "ds_b"} == {"cross_test"}
    assert {r.split for r in rows if r.emitter_id == "model_b"} == {"generalization"}


def test_class_filter_drops_labels_outside_the_set(tmp_path, capsys):
    make_fake_shards(tmp_path / "s", classes=("drone_link", "wifi", "pulsed_radar"), recordings_per_class=3)
    rows, report = build_with_report([tmp_path / "s"], seed=1, cap=0, classes=("drone_link", "wifi"))
    assert report.dropped_labels == {"pulsed_radar": 3 * 6}
    assert {r.label for r in rows} == {"drone_link", "wifi"}
    out = tmp_path / "m.jsonl"
    assert build_manifest.main(["--shards", str(tmp_path / "s"), "--out", str(out), "--seed", "1"]) == 0
    err = capsys.readouterr().err
    assert "dropped label 'pulsed_radar'" in err and "18 windows" in err
    assert {r.label for r in manifest.read(out)} == {"drone_link", "wifi"}
    assert build_manifest.main(["--shards", str(tmp_path / "s"), "--out", str(out), "--seed", "1", "--classes", "v2"]) == 0
    assert {r.label for r in manifest.read(out)} == {"drone_link", "wifi", "pulsed_radar"}


def test_burst_id_is_carried_into_the_manifest(shards, tmp_path):
    out = tmp_path / "m.jsonl"
    argv = ["--out", str(out), "--seed", "1"]
    for s in shards:
        argv += ["--shards", str(s)]
    assert build_manifest.main(argv) == 0
    rows = manifest.read(out)
    assert all(r.burst_id and r.burst_id.startswith(r.dataset + "/") for r in rows)
    assert {r.burst_id for r in rows} == {r.burst_id for r in collect_rows(shards)}


def test_holdout_by_recording_substring(shards):
    rows = collect_rows(shards)
    moved = holdout_emitters(rows, ["ds_b_wifi"], split="val", match="recording")
    assert moved == 3 * 6
    assert {r.split for r in rows if "ds_b_wifi" in r.recording_id} == {"val"}


def test_holdout_unknown_emitter_is_an_error(shards):
    with pytest.raises(ValueError, match="no rows match"):
        build(shards, seed=5, cap=0, holdout=["nope"])


def test_hash_audit_finds_identical_windows_across_splits(shards):
    rows = build(shards, seed=5, cap=0)
    store = WindowStore()
    assert audit_window_hashes(rows, store) == []
    train = next(r for r in rows if r.split == "train")
    other = next(r for r in rows if r.split != "train")
    # Point a window of another split at the same samples as a train window.
    other.shard, other.offset = train.shard, train.offset
    leaks = audit_window_hashes(rows, store)
    assert len(leaks) == 1
    assert {wid for wid, _ in leaks[0][1]} == {train.window_id, other.window_id}


def test_hash_audit_covers_the_generalization_split(shards):
    rows = build(shards, seed=5, cap=0, holdout=["model_a"])
    store = WindowStore()
    assert audit_window_hashes(rows, store) == []
    train = next(r for r in rows if r.split == "train")
    held = next(r for r in rows if r.split == "generalization")
    held.shard, held.offset = train.shard, train.offset
    (leak,) = audit_window_hashes(rows, store)
    assert sorted(split for _, split in leak[1]) == ["generalization", "train"]


def test_labels_and_weights(shards):
    rows = collect_rows(shards)
    labels = label_indices(rows, ("drone_link", "wifi"))
    assert labels.dtype == np.int64 and set(labels.tolist()) == {0, 1}
    weights = row_weights(rows)
    assert weights.dtype == np.float32
    assert set(weights.tolist()) == {1.0, 0.5}
    assert all(w == (0.5 if r.label_strength == "weak" else 1.0) for w, r in zip(weights, rows))
    with pytest.raises(ValueError, match="not one of"):
        label_indices(rows, ("wifi",))


def test_cli_writes_manifest_and_prints_digest(shards, tmp_path, capsys):
    out = tmp_path / "data" / "manifest.jsonl"
    argv = ["--out", str(out), "--seed", "7", "--cross-test", "ds_b", "--cap-per-class", "20", "--classes", "v1"]
    for s in shards:
        argv += ["--shards", str(s)]
    assert build_manifest.main(argv) == 0
    printed = capsys.readouterr().out
    assert f"sha256={manifest.digest(out)}" in printed
    rows = manifest.read(out)
    assert all(not Path(r.shard).is_absolute() for r in rows)
    windows, labels = WindowStore(out.parent).get(rows[:5], ("drone_link", "wifi"))
    assert windows.shape == (5, WINDOW_SAMPLES)
    # Same inputs and seed: byte-identical manifest.
    first = out.read_bytes()
    assert build_manifest.main(argv) == 0
    assert out.read_bytes() == first
    argv[argv.index("7")] = "8"
    assert build_manifest.main(argv) == 0
    assert out.read_bytes() != first


def test_cli_fails_on_identical_windows_across_splits(tmp_path, capsys):
    make_fake_shards(tmp_path / "s", recordings_per_class=8)
    seed = 3
    rows = build([tmp_path / "s"], seed=seed, cap=0)
    # Rewrite a test recording's rows so it reuses a train recording's shard samples.
    first = next(r for r in rows if r.split == "train")
    second = next(r for r in rows if r.split == "test")
    first.shard, second.shard = Path(first.shard).name, Path(second.shard).name
    rows_file = tmp_path / "s" / second.shard.replace(".npy", ".rows.jsonl")
    lines = []
    for line in rows_file.read_text().splitlines():
        d = json.loads(line)
        d["shard"] = first.shard
        lines.append(json.dumps(d))
    rows_file.write_text("\n".join(lines) + "\n")
    code = build_manifest.main(["--shards", str(tmp_path / "s"), "--out", str(tmp_path / "m.jsonl"), "--seed", str(seed)])
    assert code == 1
    assert "identical windows" in capsys.readouterr().err


def test_stratified_puts_simulated_recordings_in_train_only(tmp_path):
    make_fake_shards(tmp_path, recordings_per_class=10, windows_per_recording=2)
    rows = collect_rows([tmp_path])
    fake = {r.recording_id for r in rows[: len(rows) // 2]}
    for row in rows:
        row.is_real = row.recording_id not in fake
    manifest.assign_splits_stratified(rows, 3)
    assert {r.split for r in rows if not r.is_real} == {"train"}
    assert {"val", "test"} <= {r.split for r in rows if r.is_real}
