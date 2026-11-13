import hashlib
import http.server
import threading

import numpy as np
import pytest

from mcls import fetch
from mcls.sources import get
from mcls.sources.base import raw_root
from mcls.sources.tampere import Tampere, parse_name
from mcls.taxonomy import Taxonomy, load


@pytest.mark.parametrize(
    "name, expected",
    [
        ("DJI_mavic_mini_2G.bin", ("DJI_mavic_mini", "control_and_video", "2G", 1, 1)),
        ("DJI_inspire_2_5G_1of2.bin", ("DJI_inspire_2", "control_and_video", "5G", 1, 2)),
        ("Parrot_mambo_video_2G.bin", ("Parrot_mambo", "video", "2G", 1, 1)),
        ("Yuneec_typhoon_h_2G_2of2.bin", ("Yuneec_typhoon_h", "control_and_video", "2G", 2, 2)),
    ],
)
def test_parse_name(name, expected):
    p = parse_name(name)
    assert (p["emitter"], p["link"], p["band"], p["part"], p["parts"]) == expected


def test_parse_name_rejects_other_files():
    assert parse_name("load_bin.py") is None
    assert parse_name("DJI_x_3G.bin") is None


def write_tampere(folder, name, iq):
    pairs = np.empty(2 * len(iq), "<i2")
    pairs[0::2], pairs[1::2] = iq.real, iq.imag
    (folder / name).write_bytes(pairs.tobytes())


def test_loader_reads_int16_le_and_halves_share_an_id(tmp_path):
    iq = (np.arange(10) * 100 + 1j * -np.arange(10) * 50).astype(np.complex64)
    write_tampere(tmp_path, "DJI_inspire_2_5G_1of2.bin", iq)
    write_tampere(tmp_path, "DJI_inspire_2_5G_2of2.bin", iq)
    write_tampere(tmp_path, "Yuneec_typhoon_h_5G.bin", iq)
    recs = {r.source_file: r for r in Tampere(root=tmp_path).recordings()}
    a, b = recs["DJI_inspire_2_5G_1of2.bin"], recs["DJI_inspire_2_5G_2of2.bin"]
    assert a.recording_id == b.recording_id
    assert a.sample_rate_hz == 200e6 and a.center_frequency_hz == 5.8e9 and a.num_samples == 10
    assert recs["Yuneec_typhoon_h_5G.bin"].center_frequency_hz == 5.7e9
    np.testing.assert_allclose(a.read(2, 3), iq[2:5] / 32768, rtol=1e-6)
    assert a.read(0, 4).dtype == np.complex64
    with pytest.raises(IndexError):
        a.read(8, 5)


def test_check_reports_partial_downloads(tmp_path):
    write_tampere(tmp_path, "DJI_mavic_mini_2G.bin", np.zeros(10, np.complex64))
    assert any("partial download" in p for p in Tampere(root=tmp_path).check())


def test_real_taxonomy_labels_and_emitters(tmp_path):
    """Against configs/taxonomy.toml: every parsed emitter must equal the taxonomy's emitter_id."""
    iq = np.ones(4, np.complex64)
    names = ["DJI_mavic_mini_2G.bin", "DJI_phantom_4_pro_plus_5G.bin", "DJI_phantom_4_2G.bin",
             "Parrot_mambo_control_2G.bin", "Parrot_mambo_video_2G.bin", "Parrot_disco_5G.bin", "DJI_inspire_2_5G_1of2.bin"]
    for name in names:
        write_tampere(tmp_path, name, iq)
    recs = {r.source_file: r for r in Tampere(root=tmp_path, taxonomy=load()).recordings()}
    assert {f: (r.label, r.emitter_id) for f, r in recs.items()} == {
        "DJI_mavic_mini_2G.bin": ("wifi", "DJI_mavic_mini"),
        "DJI_phantom_4_pro_plus_5G.bin": ("drone_link", "DJI_phantom_4_pro_plus"),
        "DJI_phantom_4_2G.bin": ("drone_link", "DJI_phantom_4"),
        "Parrot_mambo_control_2G.bin": ("bluetooth", "Parrot_mambo"),
        "Parrot_mambo_video_2G.bin": ("wifi", "Parrot_mambo"),
        "Parrot_disco_5G.bin": ("wifi", "Parrot_disco"),
        "DJI_inspire_2_5G_1of2.bin": ("drone_link", "DJI_inspire_2"),
    }
    for f, r in recs.items():
        assert parse_name(f)["emitter"] == r.emitter_id


def test_excluded_files_are_skipped_and_emitter_falls_back(tmp_path):
    iq = np.ones(4, np.complex64)
    write_tampere(tmp_path, "DJI_mavic_mini_2G.bin", iq)
    write_tampere(tmp_path, "Parrot_disco_2G.bin", iq)
    entry = {"license": "CC BY 4.0", "is_real": True, "default_label": "wifi", "default_strength": "weak",
             "overrides": [{"match": "Parrot_disco", "exclude": True, "note": "test"}]}
    t = Taxonomy({"classes": {"v1": ["wifi"]}, "datasets": {"tampere": entry}})
    recs = list(Tampere(root=tmp_path, taxonomy=t).recordings())
    assert [(r.source_file, r.emitter_id) for r in recs] == [("DJI_mavic_mini_2G.bin", "DJI_mavic_mini")]


REAL = raw_root() / "tampere"


@pytest.mark.skipif(not (REAL / "DJI_mavic_mini_2G.bin").exists(), reason="real Tampere sample not downloaded")
def test_real_tampere_files():
    """Smoke test on the downloaded sample: sizes, rates, and the dataset's own reader logic."""
    source = get("tampere")
    assert source.check() == []
    for rec in source.recordings():
        reference = np.fromfile(source.root / rec.source_file, dtype="<i2", count=20_000).astype(np.float64)
        reference = reference.view(np.complex128) / 32768
        np.testing.assert_allclose(rec.read(0, 10_000), reference, atol=1e-7)
        expected = source.taxonomy.label_for("tampere", rec.source_file).label
        assert rec.label == expected and rec.is_real and rec.license == "CC BY 4.0"


class RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    ignore_range = False

    def do_GET(self):
        data = self.payload
        header = self.headers.get("Range")
        if header and not self.ignore_range:
            start, end = header.removeprefix("bytes=").split("-")
            start, end = int(start), int(end) if end else len(data) - 1
            body = data[start : end + 1]
            self.send_response(206)
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    handler = type("H", (RangeHandler,), {"payload": bytes(range(256)) * 4000})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.handle_error = lambda *args: None  # the no-Range test drops connections on purpose
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd, handler
    httpd.shutdown()


def test_ranges_cover_the_file_exactly():
    for size, parts in [(10, 3), (1_000_003, 4), (5, 8)]:
        spans = fetch.ranges(size, parts)
        assert spans[0][0] == 0 and spans[-1][1] == size - 1
        assert all(b[0] == a[1] + 1 for a, b in zip(spans, spans[1:]))


def test_parallel_download_verifies_and_resumes(server, tmp_path):
    httpd, handler = server
    url = f"http://127.0.0.1:{httpd.server_address[1]}/f"
    md5 = hashlib.md5(handler.payload).hexdigest()
    out = tmp_path / "f.bin"
    # A half-finished earlier run left part 0 partly written: the rerun resumes it.
    parts = out.with_name("f.bin.parts")
    parts.mkdir()
    (parts / "000").write_bytes(handler.payload[:1000])
    fetch.download(url, out, len(handler.payload), md5, connections=4, progress=False)
    assert out.read_bytes() == handler.payload and not parts.exists()


def test_bad_md5_is_refused(server, tmp_path):
    httpd, handler = server
    url = f"http://127.0.0.1:{httpd.server_address[1]}/f"
    with pytest.raises(IOError, match="md5"):
        fetch.download(url, tmp_path / "f.bin", len(handler.payload), "0" * 32, connections=2, progress=False)
    assert not (tmp_path / "f.bin").exists()


def test_server_without_range_support_is_refused(server, tmp_path, monkeypatch):
    httpd, handler = server
    handler.ignore_range = True
    monkeypatch.setattr(fetch, "RETRIES", 1)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/f"
    with pytest.raises(IOError, match="Range"):
        fetch.download(url, tmp_path / "f.bin", len(handler.payload), None, connections=2, progress=False)


def test_connection_cap():
    with pytest.raises(ValueError, match="connections"):
        fetch.download("http://x", "y", 10, connections=fetch.MAX_CONNECTIONS + 1)
