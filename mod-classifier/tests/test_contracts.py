import numpy as np
import pytest

from mcls.contracts import (
    SAMPLE_DTYPE,
    WINDOW_SAMPLES,
    Classifier,
    ManifestRow,
    Recording,
    band_of,
    check_probabilities,
    check_samples,
    check_windows,
)


def row(**overrides):
    values = dict(
        window_id="tampere/x/0", shard="shards/x.npy", offset=0, length=WINDOW_SAMPLES,
        dataset="tampere", recording_id="x", label="drone_link", label_strength="strong",
        native_rate_hz=120e6, center_frequency_hz=2.44e9, band="ism_2400", is_real=True,
        license="CC BY 4.0", split="train",
    )
    values.update(overrides)
    return ManifestRow(**values)


@pytest.mark.parametrize(
    "freq, band",
    [(2.44e9, "ism_2400"), (5.8e9, "ism_5800"), (5.3e9, "unii_5ghz"), (1575.42e6, "gnss_l1"),
     (1090e6, "adsb_1090"), (915e6, "ism_915"), (862.5e6, "other"), (866e6, "srd_868"), (98e6, "fm_broadcast")],
)
def test_band_of(freq, band):
    assert band_of(freq) == band


def test_manifest_row_round_trips_and_validates():
    original = row(snr_db_est=12.5)
    assert ManifestRow.from_dict(original.to_dict()) == original
    original.validate(("drone_link", "wifi"))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"length": 1024}, "length must be"),
        ({"split": "holdout"}, "split must be"),
        ({"label_strength": "maybe"}, "label_strength"),
        ({"license": ""}, "licence"),
        ({"snr_db_est": float("nan")}, "not finite"),
    ],
)
def test_bad_rows_are_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        row(**overrides).validate()


def test_unknown_label_and_unknown_field():
    with pytest.raises(ValueError, match="not one of"):
        row(label="ufo").validate(("drone_link",))
    with pytest.raises(ValueError, match="unknown manifest fields"):
        ManifestRow.from_dict({**row().to_dict(), "colour": "red"})


def test_recording_validate():
    rec = Recording("tampere", "x", "x.bin", 120e6, 2.44e9, 100, "drone_link", "strong", True, "CC BY 4.0",
                    read=lambda s, n: np.zeros(n, SAMPLE_DTYPE))
    rec.validate()
    assert rec.band == "ism_2400"
    rec.license = ""
    with pytest.raises(ValueError, match="licence"):
        rec.validate()


def test_sample_and_window_checks():
    check_samples(np.zeros(8, SAMPLE_DTYPE))
    with pytest.raises(ValueError):
        check_samples(np.zeros(8, np.complex128))
    bad = np.zeros(8, SAMPLE_DTYPE)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        check_samples(bad)
    check_windows(np.zeros((3, WINDOW_SAMPLES), SAMPLE_DTYPE))
    with pytest.raises(ValueError):
        check_windows(np.zeros((3, 100), SAMPLE_DTYPE))


def test_classifier_protocol_and_probabilities():
    class Uniform:
        classes = ("a", "b")

        def classify(self, windows):
            return np.full((len(windows), 2), 0.5)

    model = Uniform()
    assert isinstance(model, Classifier)
    check_probabilities(model.classify(np.zeros((4, WINDOW_SAMPLES), SAMPLE_DTYPE)), 4, model.classes)
    with pytest.raises(ValueError, match="sum to 1"):
        check_probabilities(np.full((4, 2), 0.4), 4, model.classes)
