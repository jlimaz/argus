"""Fetch short slices of real TEST-split recordings, for trying the UI (mod-classifier/ui).

    python scripts/fetch_ui_samples.py OUT_DIR

Each sample is a slice of a recording whose recording_id is in the wave 5 test
split (data/manifest_w5.jsonl), so the models never trained on it. Slices come
from the public sources by HTTP range request (no full download), except DroneRF
(one 174 MB archive from Mendeley, fast) and LoRaIQ (two members of the zip, read
by range). Writes the samples and OUT_DIR/samples.json (file, format, sample rate,
centre, true class, source, licence). About 50 MB in total.

Run on CRC (needs unrar-cffi for DroneRF). NIST Wi-Fi/Bluetooth is not included:
its captures sit in a 21.6 GB HDF5 file that cannot be sliced by range.
"""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np

UA = "mcls-ui-samples/1.0"


def fetch(url: str, start: int, length: int) -> bytes:
    return subprocess.run(["curl", "-sfL", "-A", UA, "-r", f"{start}-{start + length - 1}", url],
                          check=True, capture_output=True).stdout


def fetch_all(url: str) -> bytes:
    return subprocess.run(["curl", "-sfL", "-A", UA, url], check=True, capture_output=True).stdout


def zip_member(url: str, total: int, name: str, prefix_bytes: int | None = None) -> bytes:
    """One member of a remote zip (or the first prefix_bytes of it once inflated), by range requests."""
    tail = 64 * 1024 * 1024
    with tempfile.TemporaryDirectory() as tmp:
        sparse = Path(tmp) / "s.zip"
        with open(sparse, "wb") as f:
            f.truncate(total)
            start = max(0, total - tail)
            f.seek(start)
            f.write(fetch(url, start, total - start))
        info = zipfile.ZipFile(sparse).getinfo(name)
    header = fetch(url, info.header_offset, 30)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    data_start = info.header_offset + 30 + name_len + extra_len
    want = info.compress_size if prefix_bytes is None else min(info.compress_size, prefix_bytes)
    raw = fetch(url, data_start, want)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompressobj(-15).decompress(raw)
    return raw


def write_cs16(path: Path, iq: np.ndarray) -> None:
    scale = 32767 / max(1e-9, float(np.abs(np.concatenate([iq.real, iq.imag])).max()))
    pairs = np.empty(2 * len(iq), dtype="<i2")
    pairs[0::2] = np.round(iq.real * scale)
    pairs[1::2] = np.round(iq.imag * scale)
    pairs.tofile(path)


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    samples = []

    def add(file, fmt, rate, centre, true_class, source, licence, note=""):
        samples.append({"file": file, "format": fmt, "sample_rate_hz": rate, "center_frequency_hz": centre,
                        "true_class": true_class, "split": "test", "source": source, "licence": licence,
                        "note": note})
        print("wrote", file)

    zen = "https://zenodo.org/api/records/4264467/files/{}/content"
    # Tampere: int16 little-endian I/Q, 2.44 GHz at 120 MS/s, 5.7 GHz video at 200 MS/s (Yuneec).
    (out / "tampere_DJI_inspire_2_2G.cs16").write_bytes(fetch(zen.format("DJI_inspire_2_2G.bin"), 0, 12_000_000))
    add("tampere_DJI_inspire_2_2G.cs16", "ci16_le", 120e6, 2.44e9, "drone_link",
        "Tampere (Zenodo 4264467), DJI_inspire_2_2G.bin, first 25 ms", "CC BY 4.0")
    (out / "tampere_Yuneec_typhoon_h_5G.cs16").write_bytes(fetch(zen.format("Yuneec_typhoon_h_5G.bin"), 0, 12_000_000))
    add("tampere_Yuneec_typhoon_h_5G.cs16", "ci16_le", 200e6, 5.7e9, "drone_link",
        "Tampere (Zenodo 4264467), Yuneec_typhoon_h_5G.bin, first 15 ms", "CC BY 4.0",
        "Labelled drone_link but possibly Wi-Fi (taxonomy marks it OPEN); all models call it wifi.")

    # Triangular jammer: gzip of int8 I/Q at 10 MS/s, GPS L1. A gzip prefix inflates on its own.
    tri = ("https://zenodo.org/api/records/14859381/files/"
           "LINKS_L1_10Mhz_complex_int8_20250118_155408_TRI_80.gz/content")
    data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(fetch(tri, 0, 6_000_000))[:8_000_000]
    (out / "triangular_TRI_80_L1.cs8").write_bytes(data)
    add("triangular_TRI_80_L1.cs8", "ci8", 10e6, 1575.42e6, "gnss_jammer",
        "Triangular L1 (Zenodo 14859381), TRI_80 recording, first 0.4 s", "CC BY 4.0")

    # GATEMAN AM-tone jammer: big-endian int16 at 20 MS/s inside rawdata.zip; first 8 MB once inflated.
    gate = "https://zenodo.org/api/records/2654322/files/rawdata.zip/content"
    raw = zip_member(gate, 48748425742, "GALE1+AMtone/AMtone@20MSps-16bit.bin", prefix_bytes=12_000_000)[:8_000_000]
    be = np.frombuffer(raw[: len(raw) - len(raw) % 4], dtype=">i2").astype("<i2")
    be.tofile(out / "gateman_GALE1_AMtone.cs16")
    add("gateman_GALE1_AMtone.cs16", "ci16_le", 20e6, 1575.42e6, "gnss_jammer",
        "GATEMAN (Zenodo 2654322), GALE1+AMtone jammer file, first 0.1 s (byte order converted to little-endian)",
        "CC BY 4.0")

    # DroneRF Parrot Bebop (Wi-Fi link): real samples at 40 MS/s in CSV inside a RAR; converted as training did.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pipeline.features import real_to_iq
    from unrar.cffi import rarfile

    rar_url = ("https://data.mendeley.com/public-files/datasets/f4c2b4n755/files/"
               "8ee8e8a0-dc69-4925-af4b-bea9b03f60c8/file_downloaded")
    with tempfile.TemporaryDirectory() as tmp:
        rar_path = Path(tmp) / "RF Data_10000_H.rar"
        rar_path.write_bytes(fetch_all(rar_url))
        archive = rarfile.RarFile(str(rar_path))
        member = sorted(i.filename for i in archive.infolist() if i.filename.endswith(".csv"))[0]
        values = np.loadtxt(io.StringIO(archive.read(member).decode()), delimiter=",", dtype=np.float32)
    iq, rate, offset = real_to_iq(values.reshape(-1)[:4_000_000], 40e6)
    write_cs16(out / "dronerf_parrot_bebop_wifi.cs16", iq[:2_000_000])
    add("dronerf_parrot_bebop_wifi.cs16", "ci16_le", float(rate), 2440e6 + float(offset), "wifi",
        f"DroneRF (Mendeley f4c2b4n755), {member}, first 0.1 s after real-to-IQ", "CC BY 4.0",
        "Parrot Bebop flight link, which is Wi-Fi.")

    # rtl_433 (no licence stated; used by Odon's decision): two test-split devices.
    gh = "https://raw.githubusercontent.com/merbanan/rtl_433_tests/master/"
    for rel, centre in (("tests/GT-WT-03/01/g001_433.92M_250k.cu8", 433.92e6), ("tests/Citroen_TPMS/gfile001.cu8", 433.92e6)):
        name = "rtl433_" + rel.split("/")[1] + "_" + rel.split("/")[-1]
        (out / name).write_bytes(fetch_all(gh + rel))
        add(name, "cu8", 250e3, centre, "other_ism", f"rtl_433_tests, {rel}",
            "none stated (used by Odon's decision)", "Rate and centre from rtl_433's defaults when the name has none.")

    # LoRaIQ: one frame of a test-split session, SigMF (cf32, 250 kS/s, 862.5 MHz). Frame numbers have
    # gaps (there is no 11 in this session), so only frame 10 is fetched.
    lora = "https://zenodo.org/api/records/17708397/files/sigmfs.zip/content"
    for n in ("10",):
        for ext in ("sigmf-meta", "sigmf-data"):
            member = f"sigmfs/25_09_18-13_56/rrh1/{n}.{ext}"
            (out / f"loraiq_25_09_18-13_56_rrh1_{n}.{ext}").write_bytes(zip_member(lora, 49746561196, member))
        add(f"loraiq_25_09_18-13_56_rrh1_{n}.sigmf-data", "sigmf", 250e3, 862.5e6, "lora",
            f"LoRaIQ (Zenodo 17708397), session 25_09_18-13_56, receiver 1, frame {n}", "CC BY 4.0",
            "Upload the .sigmf-meta and .sigmf-data together.")

    (out / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
