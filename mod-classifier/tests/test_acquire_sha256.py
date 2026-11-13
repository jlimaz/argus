"""sha256 verification in mcls.fetch (parallel and single-stream); no network."""

import hashlib
import http.server
import threading

import pytest

from mcls import fetch

PAYLOAD = bytes(range(256)) * 1000


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        header = self.headers.get("Range")
        if header:
            start, end = header.removeprefix("bytes=").split("-")
            body = PAYLOAD[int(start) : int(end) + 1]
            self.send_response(206)
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def url():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/f"
    httpd.shutdown()


GOOD = hashlib.sha256(PAYLOAD).hexdigest()


def test_sha256_passes(url, tmp_path):
    out = fetch.download(url, tmp_path / "f", len(PAYLOAD), sha256=GOOD, connections=3, progress=False)
    assert out.read_bytes() == PAYLOAD


def test_sha256_mismatch_is_refused(url, tmp_path):
    with pytest.raises(IOError, match="sha256"):
        fetch.download(url, tmp_path / "f", len(PAYLOAD), sha256="0" * 64, connections=2, progress=False)
    assert not (tmp_path / "f").exists()


def test_sha256_wins_over_md5(url, tmp_path):
    with pytest.raises(IOError, match="sha256"):
        fetch.download(url, tmp_path / "f", len(PAYLOAD), md5=hashlib.md5(PAYLOAD).hexdigest(), sha256="1" * 64,
                       connections=1, progress=False)


def test_existing_good_file_is_kept(url, tmp_path):
    (tmp_path / "f").write_bytes(PAYLOAD)
    assert fetch.download("http://127.0.0.1:1/unreachable", tmp_path / "f", len(PAYLOAD), sha256=GOOD,
                          progress=False).read_bytes() == PAYLOAD


def test_stream_download_sha256(url, tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    assert fetch.stream_download(url, tmp_path / "f", len(PAYLOAD), sha256=GOOD, progress=False).read_bytes() == PAYLOAD


def test_stream_download_mismatch_keeps_bad_file(url, tmp_path):
    """No resume on single-stream hosts: a mismatched download is kept as <name>.bad, not deleted."""
    with pytest.raises(IOError, match=r"sha256 .* kept as .*g\.bad .*not deleted"):
        fetch.stream_download(url, tmp_path / "g", len(PAYLOAD), sha256="2" * 64, progress=False)
    assert not (tmp_path / "g").exists() and not (tmp_path / "g.tmp").exists()
    assert (tmp_path / "g.bad").read_bytes() == PAYLOAD


@pytest.mark.parametrize(
    "text, ok",
    [
        ("sha256:" + "a" * 64, True),
        ("md5:" + "B" * 32, True),
        ("sha256:" + "a" * 63, False),
        ("sha384:" + "a" * 96, False),
        ("a" * 64, False),
        ("md5:" + "g" * 32, False),
    ],
)
def test_parse_checksum(text, ok):
    if ok:
        algorithm, value = fetch.parse_checksum(text)
        assert value == value.lower() and algorithm in ("md5", "sha256")
    else:
        with pytest.raises(ValueError):
            fetch.parse_checksum(text)
