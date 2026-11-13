"""Schema rules of mcls.taxonomy, on small test-built taxonomies."""

import pytest

from mcls.taxonomy import EXCLUDED, Label, Taxonomy


def make(**dataset):
    entry = {"license": "CC BY 4.0", "is_real": True, "default_label": "a", "default_strength": "strong"}
    entry.update(dataset)
    return Taxonomy({"classes": {"v1": ["a", "b"]}, "datasets": {"d": entry}})


def test_label_without_new_fields_keeps_old_shape():
    label = Label("a", "strong")
    assert label.note is None and label.emitter_id is None and label.excluded is False
    assert Label("a", "weak", "n") == Label("a", "weak", note="n")


def test_emitter_id_comes_from_the_override():
    tax = make(overrides=[{"match": "x", "label": "b", "strength": "weak", "emitter_id": "drone_x"}])
    assert tax.label_for("d", "file_x.bin") == Label("b", "weak", None, "drone_x")
    assert tax.label_for("d", "file_y.bin").emitter_id is None


def test_first_match_wins():
    tax = make(overrides=[
        {"match": "abc", "label": "b", "strength": "strong", "emitter_id": "long"},
        {"match": "ab", "label": "a", "strength": "weak", "emitter_id": "short"},
    ])
    assert tax.label_for("d", "abc.bin").emitter_id == "long"
    assert tax.label_for("d", "abd.bin").emitter_id == "short"


def test_excluded_override_returns_an_excluded_label():
    tax = make(overrides=[{"match": "gnss", "exclude": True, "note": "no class"}])
    label = tax.label_for("d", "gnss.bin")
    assert label.excluded and label.label == EXCLUDED and label.note == "no class"
    assert label.label not in tax.all_classes


def test_excluded_default():
    tax = make(default_exclude=True, note="unknown files", default_label=None, default_strength=None)
    assert tax.label_for("d", "anything").excluded


def test_excluded_default_needs_no_default_label():
    tax = Taxonomy({"classes": {"v1": ["a"]}, "datasets": {"d": {
        "license": "x", "is_real": True, "default_exclude": True, "note": "why"}}})
    assert tax.label_for("d", "f").excluded


@pytest.mark.parametrize("bad, message", [
    ({"overrides": [{"match": "x", "exclude": True}]}, "without a note"),
    ({"default_exclude": True}, "without a note"),
    ({"overrides": [{"match": "x", "label": "b"}]}, "missing strength"),
    ({"overrides": [{"label": "b", "strength": "weak"}]}, "without `match`"),
    ({"overrides": [{"match": "x", "label": "zzz", "strength": "weak"}]}, "unknown class"),
    ({"overrides": [{"match": "x", "label": "b", "strength": "maybe"}]}, "unknown strength"),
])
def test_schema_errors(bad, message):
    with pytest.raises(ValueError, match=message):
        make(**bad)


def test_missing_default_label_is_an_error():
    with pytest.raises(ValueError, match="missing default_label"):
        Taxonomy({"classes": {"v1": ["a"]}, "datasets": {"d": {"license": "x", "is_real": True}}})


def test_excluded_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        Taxonomy({"classes": {"v1": [EXCLUDED]}})
