"""progress.png helpers: history reading and the summary line need no torch and no matplotlib."""

import importlib.util
import json

import pytest

from mcls.train import progress


def write_history(run_dir, rows, extra=""):
    run_dir.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows) + extra
    (run_dir / "history.jsonl").write_text(text)


def row(epoch, train, val, f1):
    return {"epoch": epoch, "train_loss": train, "val_loss": val, "val_macro_f1": f1}


def test_read_history_sorts_and_skips_half_written_line(tmp_path):
    run = tmp_path / "run"
    write_history(run, [row(1, 0.5, 0.9, 0.4), row(0, 1.2, 1.3, 0.1)], extra='{"epoch": 2, "train_')
    assert [r["epoch"] for r in progress.read_history(run)] == [0, 1]


def test_read_history_without_file_is_empty(tmp_path):
    assert progress.read_history(tmp_path) == []
    assert progress.write_plot(tmp_path) is None
    assert progress.summary([]) == "no epochs yet"


def test_summary_names_last_and_best_epoch():
    rows = [row(0, 1.2, 1.3, 0.10), row(1, 0.8, 0.9, 0.62), row(2, 0.5, 1.0, 0.55)]
    text = progress.summary(rows)
    assert text.startswith("epoch 2: train loss 0.500, val loss 1.000, val macro-F1 0.550")
    assert "best 0.620 at epoch 1" in text


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None, reason="matplotlib not installed (CRC only)")
def test_write_plot_makes_a_png(tmp_path):
    run = tmp_path / "run"
    write_history(run, [row(0, 1.2, 1.3, 0.10), row(1, 0.8, 0.9, 0.62)])
    path = progress.write_plot(run, total_epochs=40)
    assert path == run / progress.PLOT_NAME
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (run / (progress.PLOT_NAME + ".tmp")).exists()
    assert (run / "progress" / "epoch_001.png").read_bytes() == path.read_bytes()


def test_write_plot_without_matplotlib_returns_none(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    run = tmp_path / "run"
    write_history(run, [row(0, 1.2, 1.3, 0.10)])
    assert progress.write_plot(run) is None
