"""mcls.acquire against a local HTTP server and a test catalog; no network."""

import hashlib
import http.server
import json
import threading

import pytest

from mcls import acquire, fetch

PAYLOAD_A = bytes(range(256)) * 400
PAYLOAD_B = bytes(reversed(range(256))) * 300


class Handler(http.server.BaseHTTPRequestHandler):
    files: dict = {}
    ignore_range = False
    requests: list = []
    tokens_used: set = set()

    def do_GET(self):
        self.requests.append((self.path, self.headers.get("Range")))
        path, _, query = self.path.partition("?")
        if query.startswith("token="):  # Fairdata-like: single-use token, no ranges
            token = query.removeprefix("token=")
            if token in self.tokens_used:
                self.send_response(401)
                self.end_headers()
                return
            self.tokens_used.add(token)
        data = self.files.get(path)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
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
    handler = type(
        "H", (Handler,), {"files": {"/a.bin": PAYLOAD_A, "/b.bin": PAYLOAD_B}, "requests": [], "tokens_used": set()}
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    httpd.handle_error = lambda *args: None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    httpd.shutdown()


def write_catalog(path, base_url, sha_a=None):
    sha_a = sha_a or hashlib.sha256(PAYLOAD_A).hexdigest()
    md5_b = hashlib.md5(PAYLOAD_B).hexdigest()
    text = f"""
[hosts.local]
max_connections = 8
supports_range = true
note = "test"

[hosts.fairdata]
max_connections = 1
supports_range = false
note = "test"

[hosts.ieee_dataport]
max_connections = 1
supports_range = false
note = "test"

[datasets.demo]
title = "Demo"
page = "{base_url}/"
host = "local"
access = "direct"
licence = "CC0"
files_status = "complete"
files_note = "test"

[[datasets.demo.files]]
name = "sub/dir/a.bin"
size = {len(PAYLOAD_A)}
checksum = "sha256:{sha_a}"
url = "{base_url}/a.bin"

[[datasets.demo.files]]
name = "b.bin"
role = "metadata"
size = {len(PAYLOAD_B)}
checksum = "md5:{md5_b}"
url = "{base_url}/b.bin"

[datasets.fd]
title = "Fairdata-like"
page = "{base_url}/"
host = "fairdata"
access = "fairdata"
licence = "CC BY 4.0"
record = "abc"
files_status = "complete"
files_note = "test"

[[datasets.fd.files]]
name = "/x/a.bin"
size = {len(PAYLOAD_A)}
checksum = "sha256:{hashlib.sha256(PAYLOAD_A).hexdigest()}"
url = "fairdata:/a.bin"

[datasets.locked]
title = "Locked"
page = "https://example.org/locked"
host = "ieee_dataport"
access = "ieee_login"
licence = "CC BY 4.0"
files_status = "unknown"
files_note = "Needs a login; resolve by logging in and listing the files."
files = []
"""
    path.write_text(text)
    return path


@pytest.fixture
def env(server, tmp_path, monkeypatch):
    base_url, handler = server
    catalog = write_catalog(tmp_path / "catalog.toml", base_url)
    raw = tmp_path / "raw"
    monkeypatch.setenv("MCLS_RAW", str(raw))
    monkeypatch.setattr(fetch, "RETRIES", 2)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    return catalog, raw, handler, base_url


def run(catalog, *args):
    return acquire.main(["--catalog", str(catalog), *args])


def test_list_prints_files_and_sizes(env, capsys):
    catalog, raw, handler, _ = env
    assert run(catalog, "--dataset", "demo", "--list") == 0
    out = capsys.readouterr().out
    assert "sub/dir/a.bin" in out and "b.bin" in out and "in 2 files" in out
    lines = {line.split()[-1]: line.split()[2] for line in out.splitlines() if " MB " in line}
    assert lines == {"sub/dir/a.bin": "data", "b.bin": "metadata"}
    assert handler.requests == [] and not raw.exists()


def test_only_fetches_one_file_into_mcls_raw_directly(env):
    catalog, raw, handler, _ = env
    assert run(catalog, "--dataset", "demo", "--only", "b.bin", "--connections", "3") == 0
    assert sorted(p.name for p in raw.iterdir()) == ["b.bin"]
    assert (raw / "b.bin").read_bytes() == PAYLOAD_B
    assert {path for path, _ in handler.requests} == {"/b.bin"}


def test_only_accepts_full_name_and_lands_by_base_name(env):
    catalog, raw, _, _ = env
    assert run(catalog, "--dataset", "demo", "--only", "sub/dir/a.bin") == 0
    assert (raw / "a.bin").read_bytes() == PAYLOAD_A and not (raw / "sub").exists()


def test_without_mcls_raw_files_go_to_data_raw_dataset(env, monkeypatch, tmp_path):
    monkeypatch.delenv("MCLS_RAW")
    monkeypatch.setattr(acquire, "DEFAULT_RAW", tmp_path / "default")
    assert acquire.dest_dir("demo") == tmp_path / "default" / "demo"
    catalog = env[0]
    assert run(catalog, "--dataset", "demo", "--only", "b.bin") == 0
    assert (tmp_path / "default" / "demo" / "b.bin").read_bytes() == PAYLOAD_B


def test_unknown_only_name_is_refused(env, capsys):
    catalog, raw, handler, _ = env
    assert run(catalog, "--dataset", "demo", "--only", "nope.bin") == 2
    assert "not in the catalog" in capsys.readouterr().err and handler.requests == []


def test_max_gb_refuses_and_fetches_nothing(env, capsys):
    catalog, raw, handler, _ = env
    assert run(catalog, "--dataset", "demo", "--max-gb", "0.00001") == 2
    assert "over --max-gb" in capsys.readouterr().err
    assert handler.requests == [] and not raw.exists()
    assert run(catalog, "--dataset", "demo", "--max-gb", "1") == 0


def test_dry_run_downloads_nothing(env, capsys):
    catalog, raw, handler, _ = env
    assert run(catalog, "--dataset", "demo", "--dry-run", "--connections", "4") == 0
    out = capsys.readouterr().out
    assert "would fetch sub/dir/a.bin" in out and str(raw / "a.bin") in out and "nothing downloaded" in out
    assert handler.requests == [] and not raw.exists()


def test_connections_are_capped_at_eight(env, capsys):
    catalog = env[0]
    assert run(catalog, "--dataset", "demo", "--dry-run", "--connections", "20") == 0
    assert "over 8 connection(s)" in capsys.readouterr().out


def test_ieee_login_dataset_fails_clearly(env, capsys):
    catalog, raw, handler, _ = env
    assert run(catalog, "--dataset", "locked") == 2
    err = capsys.readouterr().err
    assert "not direct" in err and "ieee_login" in err and "example.org/locked" in err
    assert handler.requests == []


def test_unknown_dataset_is_refused(env, capsys):
    assert run(env[0], "--dataset", "nope") == 2
    assert "known: demo" in capsys.readouterr().err


def test_fairdata_is_forced_to_one_connection(env, capsys, monkeypatch):
    catalog, raw, handler, base_url = env
    handler.ignore_range = True
    tokens = iter(range(100))
    monkeypatch.setattr(acquire, "fairdata_url", lambda record, path: f"{base_url}{path}?token={next(tokens)}")
    assert run(catalog, "--dataset", "fd", "--connections", "4") == 0
    assert "using 1, not 4" in capsys.readouterr().out
    assert (raw / "a.bin").read_bytes() == PAYLOAD_A
    assert handler.requests == [("/a.bin?token=0", None)]  # one GET, no Range header


def test_fairdata_gets_a_fresh_token_per_attempt(env, monkeypatch):
    """A dropped stream is retried from byte 0 with a new token (tokens work once)."""
    catalog, raw, handler, base_url = env
    handler.ignore_range = True
    handler.files["/short.bin"] = PAYLOAD_A[:100]
    urls = iter([f"{base_url}/short.bin?token=t1", f"{base_url}/a.bin?token=t2"])
    monkeypatch.setattr(acquire, "fairdata_url", lambda record, path: next(urls))
    assert run(catalog, "--dataset", "fd") == 0
    assert (raw / "a.bin").read_bytes() == PAYLOAD_A
    assert [p for p, _ in handler.requests] == ["/short.bin?token=t1", "/a.bin?token=t2"]


def test_fairdata_effective_connections_from_host_entry():
    assert acquire.effective_connections(4, {"max_connections": 1, "supports_range": False}) == 1
    assert acquire.effective_connections(4, {"max_connections": 8, "supports_range": False}) == 1
    assert acquire.effective_connections(4, {"max_connections": 8, "supports_range": True}) == 4


def test_real_catalog_fairdata_reduced_to_one(capsys):
    assert acquire.main(["--dataset", "fgi_jammer2023", "--connections", "4", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "using 1, not 4" in out and "over 1 connection(s)" in out


def test_fairdata_url_posts_to_etsin(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps({"url": "https://download.example/download?token=T"}).encode()

    def fake_urlopen(request, timeout):
        seen["url"], seen["body"], seen["method"] = request.full_url, json.loads(request.data), request.get_method()
        return Response()

    monkeypatch.setattr(acquire.urllib.request, "urlopen", fake_urlopen)
    assert acquire.fairdata_url("abc", "/x/a.bin") == "https://download.example/download?token=T"
    assert seen == {"url": acquire.FAIRDATA_AUTHORIZE, "body": {"cr_id": "abc", "file": "/x/a.bin"}, "method": "POST"}
