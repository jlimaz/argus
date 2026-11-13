"""The shipped configs/taxonomy.toml: every dataset entry and the drone link decisions."""

import tomllib

import pytest

from mcls import taxonomy
from mcls.taxonomy import DEFAULT_PATH, EXCLUDED

TAX = taxonomy.load()
with open(DEFAULT_PATH, "rb") as handle:
    RAW = tomllib.load(handle)["datasets"]

#: Names fixed by loaders, plus the ones engineer 1 chose.
DATASETS = (
    "tampere", "dronerf", "nist_wifi_bt", "occupancy", "dronedetect", "fgi_jammer",
    "gateman", "triangular_l1", "loraiq", "drexel_cbrs_radar", "adsb", "nist_radar", "rtl_433",
)
SIMULATED = {"drexel_cbrs_radar", "nist_radar"}


def test_every_dataset_is_present():
    assert set(TAX.dataset_names) == set(DATASETS)


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_entry_loads(name):
    info = TAX.dataset(name)
    assert info.license and info.url.startswith("https://")
    assert info.is_real is (name not in SIMULATED)
    entry = RAW[name]
    assert entry.get("default_exclude") or entry["default_label"] in TAX.all_classes


@pytest.mark.parametrize("name", DATASETS)
def test_every_override_uses_a_known_class_or_is_excluded(name):
    for override in RAW[name].get("overrides", []):
        if override.get("exclude"):
            assert override["note"]
        else:
            assert override["label"] in TAX.all_classes
            assert override["strength"] in ("strong", "weak")


# Tampere: every file in https://zenodo.org/api/records/4264467 -> (label, emitter_id).
TAMPERE = {
    "DJI_inspire_2_2G.bin": ("drone_link", "DJI_inspire_2"),
    "DJI_inspire_2_5G_1of2.bin": ("drone_link", "DJI_inspire_2"),
    "DJI_inspire_2_5G_2of2.bin": ("drone_link", "DJI_inspire_2"),
    "DJI_matrice_100_2G.bin": ("drone_link", "DJI_matrice_100"),
    "DJI_matrice_210_2G.bin": ("drone_link", "DJI_matrice_210"),
    "DJI_matrice_210_5G_1of2.bin": ("drone_link", "DJI_matrice_210"),
    "DJI_matrice_210_5G_2of2.bin": ("drone_link", "DJI_matrice_210"),
    "DJI_mavic_mini_2G.bin": ("wifi", "DJI_mavic_mini"),
    "DJI_mavic_pro_2G.bin": ("drone_link", "DJI_mavic_pro"),
    "DJI_phantom_4_2G.bin": ("drone_link", "DJI_phantom_4"),
    "DJI_phantom_4_pro_plus_2G.bin": ("drone_link", "DJI_phantom_4_pro_plus"),
    "DJI_phantom_4_pro_plus_5G_1of2.bin": ("drone_link", "DJI_phantom_4_pro_plus"),
    "DJI_phantom_4_pro_plus_5G_2of2.bin": ("drone_link", "DJI_phantom_4_pro_plus"),
    "Parrot_disco_2G.bin": ("wifi", "Parrot_disco"),
    "Parrot_mambo_control_2G.bin": ("bluetooth", "Parrot_mambo"),
    "Parrot_mambo_video_2G.bin": ("wifi", "Parrot_mambo"),
    "Yuneec_typhoon_h_2G_1of2.bin": ("drone_link", "Yuneec_typhoon_h"),
    "Yuneec_typhoon_h_2G_2of2.bin": ("drone_link", "Yuneec_typhoon_h"),
    "Yuneec_typhoon_h_5G.bin": ("drone_link", "Yuneec_typhoon_h"),
}


@pytest.mark.parametrize("file_name, expected", sorted(TAMPERE.items()))
def test_tampere_files(file_name, expected):
    label = TAX.label_for("tampere", file_name)
    assert (label.label, label.emitter_id) == expected
    assert not label.excluded


def test_tampere_emitter_ids_match_the_loader():
    from mcls.sources.tampere import parse_name

    for file_name in TAMPERE:
        assert TAX.label_for("tampere", file_name).emitter_id == parse_name(file_name)["emitter"]


def test_tampere_first_match_order_separates_phantom_4_models():
    # "DJI_phantom_4_" is a substring of the Pro Plus name; the Pro Plus override comes first.
    assert "DJI_phantom_4_" in "DJI_phantom_4_pro_plus_2G.bin"
    assert TAX.label_for("tampere", "DJI_phantom_4_pro_plus_2G.bin").emitter_id == "DJI_phantom_4_pro_plus"
    assert TAX.label_for("tampere", "DJI_phantom_4_2G.bin").emitter_id == "DJI_phantom_4"


# DroneRF BUI codes (data paper, PMC6727013) -> (label, emitter_id).
DRONERF = {"00000": ("background", None)}
DRONERF.update({f"100{m}": ("wifi", "Parrot_Bebop") for m in ("00", "01", "10", "11")})
DRONERF.update({f"101{m}": ("wifi", "Parrot_AR_Drone_2") for m in ("00", "01", "10", "11")})
DRONERF.update({f"110{m}": ("wifi", "DJI_Phantom_3_Standard") for m in ("00", "01", "10", "11")})


@pytest.mark.parametrize("bui, expected", sorted(DRONERF.items()))
@pytest.mark.parametrize("half", ["L", "H"])
@pytest.mark.parametrize("segment", [0, 3, 10, 20])
def test_dronerf_files(bui, expected, half, segment):
    label = TAX.label_for("dronerf", f"{bui}{half}_{segment}.csv")
    assert (label.label, label.emitter_id) == expected


def test_dronerf_unknown_code_is_excluded():
    assert TAX.label_for("dronerf", "README.txt").excluded


def test_gateman_files():
    for gnss in ("GALE1", "GPSL1", "GPSL5"):
        for jammer in ("AMtone", "Chirp10MHz", "Chirp20MHz"):
            folder = f"rawdata/{gnss}+{jammer}"
            assert TAX.label_for("gateman", f"{folder}/{jammer}@20MSps.bin").label == "gnss_jammer"
            gnss_label = TAX.label_for("gateman", f"{folder}/{gnss}@20MSps.bin")
            assert gnss_label.excluded and gnss_label.label == EXCLUDED


def test_fgi_spoofing_is_excluded_and_jamming_is_labelled():
    assert TAX.label_for("fgi_jammer", "JammerTest2023/JT23_4.1.5/data.bin").label == "gnss_jammer"
    assert TAX.label_for("fgi_jammer", "JammerTest2023/JT23_16.1.1/data.bin").excluded
    assert TAX.label_for("fgi_jammer", "JammerTest2023/JT23_17.1.6/data.bin").excluded


def test_other_defaults():
    assert TAX.label_for("triangular_l1", "LINKS_L1_10Mhz_complex_int8_20250118_161623_TRI_70.gz").label == "gnss_jammer"
    assert TAX.label_for("loraiq", "RRH1/1.sigmf-data").label == "lora"
    assert TAX.label_for("nist_wifi_bt", "5ghz_outdoor.h5").label == "wifi"
    assert TAX.label_for("dronedetect", "MIN_1100_00.dat").emitter_id == "DJI_mavic_mini"
    assert TAX.label_for("dronedetect", "XYZ_0000_00.dat").excluded
    # Included by Odon's decision on Nov 12, though the repository states no licence.
    assert TAX.label_for("rtl_433", "tests/6sc2/01/g001_433.92M_250k.cu8").label == "other_ism"
