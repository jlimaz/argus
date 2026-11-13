"""The real configs/catalog.toml parses and every entry carries what mcls.acquire needs."""

import pytest

from mcls import acquire, fetch
from mcls.sources import base

LOADER_NAMES = ["tampere", "dronerf", "nist_wifi_bt", "loraiq", "occupancy", "gateman", "triangular_l1"]


@pytest.fixture(scope="module")
def catalog():
    return acquire.load_catalog()


def test_loader_names_are_in_the_catalog(catalog):
    assert set(LOADER_NAMES) <= set(catalog["datasets"])


def test_every_entry_has_the_required_fields(catalog):
    for name, ds in catalog["datasets"].items():
        for key in acquire.DATASET_KEYS:
            assert ds[key], (name, key)
        assert ds["access"] in acquire.ACCESS_TYPES
        assert ds["host"] in catalog["hosts"]
        if ds["files_status"] == "unknown":
            assert ds["files"] == [] and len(ds["files_note"]) > 40, name
        for f in ds["files"]:
            assert isinstance(f["size"], int) and f["size"] > 0
            algorithm, value = fetch.parse_checksum(f["checksum"])
            assert algorithm in ("md5", "sha256")
            assert f["url"].startswith("https://") or (ds["access"] == "fairdata" and f["url"].startswith("fairdata:/"))


def test_non_direct_sets_are_marked(catalog):
    ds = catalog["datasets"]
    for name in ("occupancy", "dronedetect", "adsb_24h"):
        assert ds[name]["access"] == "ieee_login" and ds[name]["files"] == []
    assert ds["fgi_jammer2023"]["access"] == "fairdata"
    assert catalog["hosts"]["fairdata"] == {**catalog["hosts"]["fairdata"], "max_connections": 1, "supports_range": False}


def test_known_values_from_the_apis(catalog):
    """Spot checks against the API responses read on 2026-11-12 (and the md5 verified on the laptop)."""
    tampere = {f["name"]: f for f in catalog["datasets"]["tampere"]["files"]}
    assert tampere["DJI_mavic_mini_2G.bin"]["size"] == 480_000_000
    assert tampere["DJI_mavic_mini_2G.bin"]["checksum"] == "md5:5d958277d2fbafe9ee4dcdce3b4636e1"
    assert len(tampere) == 25
    assert len(catalog["datasets"]["dronerf"]["files"]) == 23
    nist = {f["name"]: f for f in catalog["datasets"]["nist_wifi_bt"]["files"]}
    assert nist["2_4ghz_indoor.h5"]["size"] == 21_600_072_448
    assert sum(n.endswith(".h5") for n in nist) == 5


def test_roles(catalog):
    ds = catalog["datasets"]

    def roles(name):
        return {f["name"]: f.get("role", "data") for f in ds[name]["files"]}

    for name, d in ds.items():
        assert all(f.get("role", "data") in acquire.ROLES for f in d["files"]), name
    nist = roles("nist_wifi_bt")
    assert sorted(n for n, r in nist.items() if r == "metadata") == sorted(
        [f"{b}.csv" for b in ("2_4ghz_bluetooth", "2_4ghz_indoor", "2_4ghz_outdoor", "5ghz_indoor", "5ghz_outdoor")]
        + ["gain_calibration_data.csv", "noise_floor_data.csv"]
    )
    assert all(r == "data" for n, r in nist.items() if n.endswith(".h5"))
    assert roles("loraiq")["dataset.csv"] == "metadata" and roles("loraiq")["sigmfs.zip"] == "data"
    assert roles("drexel_cbrs_radar") == {"RF-dataset-collection-MATLAB-scripts.tar": "doc", "Group6_data.tar": "data"}
    tampere = roles("tampere")
    assert {n for n, r in tampere.items() if r == "doc"} == {
        "load_bin.py", "load_bin.m", "example.py", "example.m",
        "example_spectrogram_Python.png", "example_spectrogram_Matlab.jpg",
    }
    assert sum(r == "data" for r in tampere.values()) == 19
    assert roles("gateman") == {"rawdata.zip": "data"}  # the record holds only the archive
    assert all(r == "data" for r in roles("dronerf").values())


def test_bad_role_is_refused(tmp_path):
    bad = tmp_path / "c.toml"
    bad.write_text(
        '[hosts.h]\nmax_connections = 1\nsupports_range = true\nnote = ""\n'
        '[datasets.d]\ntitle = "t"\npage = "p"\nhost = "h"\naccess = "direct"\nlicence = "l"\n'
        'files_status = "complete"\nfiles_note = "n"\n'
        '[[datasets.d.files]]\nname = "a"\nsize = 1\nchecksum = "md5:' + "0" * 32 + '"\nurl = "https://x"\nrole = "readme"\n'
    )
    with pytest.raises(ValueError, match="role"):
        acquire.load_catalog(bad)


def test_default_raw_matches_the_loaders():
    assert acquire.DEFAULT_RAW == base.DEFAULT_RAW


def test_bad_catalog_is_refused(tmp_path):
    bad = tmp_path / "c.toml"
    bad.write_text(
        '[hosts.h]\nmax_connections = 1\nsupports_range = true\nnote = ""\n'
        '[datasets.d]\ntitle = "t"\npage = "p"\nhost = "h"\naccess = "direct"\nlicence = "l"\n'
        'files_status = "complete"\nfiles_note = "n"\n'
        '[[datasets.d.files]]\nname = "a"\nsize = 1\nchecksum = "crc32:00"\nurl = "https://x"\n'
    )
    with pytest.raises(ValueError, match="checksum"):
        acquire.load_catalog(bad)
