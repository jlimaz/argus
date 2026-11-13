from finetune.sources.dronerf import DRONES, MODES, parse_name


def test_file_names_decode_to_bui_band_and_segment():
    assert parse_name("10111H_3.csv") == ("10111", "H", 3)
    assert parse_name("00000L_12.csv") == ("00000", "L", 12)
    assert parse_name("readme.txt") is None


def test_every_drone_bui_in_the_dataset_is_known():
    for bui in ("10000", "10001", "10010", "10011", "10100", "10101", "10110", "10111", "11000"):
        assert bui[1:3] in DRONES and bui[3:5] in MODES
