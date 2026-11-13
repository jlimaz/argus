# Data acquisition

Every dataset file the plan uses is listed in [`configs/catalog.toml`](configs/catalog.toml) with its URL, size in bytes and checksum, copied from each host's own API on 2026-11-12. `python -m mcls.acquire` fetches files from that list and refuses anything that is not listed, too big, or not openly downloadable. **Every real download still needs approval first** ([docs/history/HANDOFF.md](docs/history/HANDOFF.md) section 1).

## Use

Run from `mod-classifier/`.

```bash
python -m mcls.acquire --dataset tampere --list                      # files, sizes, total; fetches nothing
python -m mcls.acquire --dataset tampere --only DJI_mavic_mini_2G.bin --dry-run
python -m mcls.acquire --dataset tampere --only DJI_mavic_mini_2G.bin --connections 4
python -m mcls.acquire --dataset nist_wifi_bt --only 2_4ghz_indoor.h5 --max-gb 25
```

| Flag | Meaning |
| --- | --- |
| `--dataset D` | Dataset name in the catalog (required) |
| `--list` | Print every file with its size, then the total |
| `--only NAME` | Fetch only this file; repeatable. NAME is the name `--list` prints, or its base name |
| `--connections N` | Parallel byte ranges per file. Default 1, never more than 8, and never more than the host allows (Fairdata: 1) |
| `--max-gb X` | Refuse if the selected files add up to more than X GB (1 GB = 10^9 bytes) |
| `--dry-run` | Print each file, its size, where it would go and over how many connections; download nothing |

**Where files land:** in `$MCLS_RAW` itself when that variable is set (a CRC task's node `/tmp`, see `crc/prepare_array.job`), otherwise in `data/raw/<dataset>/`. Only the file's base name is kept, so NIST radar's `SimulatedRadarWaveforms/Group1/group1_subset_2.mat` lands as `group1_subset_2.mat` (base names are unique within each dataset; a test checks this).

**Roles:** each catalog file has an optional `role`: `data` (a recording, the default), `metadata` (label tables, calibration and noise-floor CSVs, LoRaIQ's `dataset.csv`) or `doc` (readers, scripts, figures, instructions). `--list` prints it. `scripts/make_file_list.py` leaves out `doc` files.

**Checks:** each file must match the catalog's size and checksum (md5 for Zenodo, sha256 for NIST, Mendeley and Fairdata). A ranged download that fails is deleted and refetched from its parts. A single-stream download (Fairdata) that fails is kept as `<name>.bad` with a message, not deleted, because it cannot resume and fetching it again costs the whole file (up to 72 GB). A file already present and verified is not fetched again. An interrupted ranged download resumes from its part files.

**Exit codes:** 0 done; 1 download or checksum failed; 2 refused (unknown dataset or file, access not direct, file list unknown, over `--max-gb`).

## Datasets

| Name | Host | Access | Files | Total |
| --- | --- | --- | --- | --- |
| `tampere` | Zenodo 4264467 | direct | 25 | 8.56 GB |
| `dronerf` | Mendeley f4c2b4n755 v1 | direct | 23 | 4.03 GB |
| `nist_wifi_bt` | NIST mds2-2731 | direct | 13 | 108.0 GB |
| `gateman` | Zenodo 2654322 | direct | 1 | 48.75 GB |
| `triangular_l1` | Zenodo 14859381 | direct | 5 | 49.88 GB |
| `loraiq` | Zenodo 17708397 | direct | 5 | 49.77 GB |
| `drexel_cbrs_radar` | Zenodo 4521678 | direct | 2 | 34.09 GB |
| `nist_radar` | NIST mds2-2116 | direct | 410 | 492.76 GB |
| `fgi_jammer2023` | Fairdata | fairdata | 3 | 151.67 GB |
| `occupancy`, `dronedetect`, `adsb_24h` | IEEE DataPort | ieee_login | unknown | unknown |
| `panoradio_hf`, `rtl_433_tests` | own site, GitHub | manual | unknown | unknown |

Notes:

- **DroneRF:** the Mendeley API returns an empty list for `folder_id=root`, which is why the list looked missing before. The dataset's folders (`/public-api/datasets/f4c2b4n755/folders/1`) each list their files with size and sha256, so the list is complete: 23 RAR files.
- **NIST:** the per-file `.sha256` sidecars are left out; their content is already in the catalog.
- **IEEE DataPort:** each page shows "LOGIN TO ACCESS DATASET FILES" (checked 2026-11-12), so names, sizes and checksums cannot be read without an account. To resolve: someone with an IEEE account copies the file list into the catalog and downloads by hand.
- **Panoradio HF** (licence unconfirmed) and **rtl_433_tests** (a git repository, to be cloned at a pinned commit) were not opened for the catalog.

## Fairdata (FGI Jammer Test 2023)

Fairdata works differently from the other hosts, and `mcls.acquire` handles it separately:

1. **Range requests are ignored.** The download service's `/download` endpoint streams the whole file from byte 0 and never reads a `Range` header ([`download/blueprints/download_api.py`](https://github.com/CSCfi/fairdata-download/blob/12c7c80516cd3c946ed581185cc4b39ccd739243/download/blueprints/download_api.py), `stream_response`, commit `12c7c80`). Checked on 2026-11-12: a GET with `Range: bytes=0-1023` got `HTTP 200`, not 206 (the transfer was stopped after 1 KB).
2. **Tokens are single-use.** The same source rejects a token already recorded in its database with 401, and its docstring calls them "single-use". Checked: a second GET with the same token got `HTTP 401`.
3. **Tokens come from Etsin.** The download service's own `/authorize` is internal. The public route is Etsin's `POST https://etsin.fairdata.fi/api/v3/download/authorize` with `{"cr_id": <dataset id>, "file": <path>}`, which returns `{"url": "https://download.fairdata.fi:443/download?token=..."}` ([`etsin_finder/resources/download_resources_common.py`](https://github.com/CSCfi/fairdata-etsin-qvain/blob/HEAD/etsin_finder/resources/download_resources_common.py), class `Authorize`; route in `etsin_finder/app.py`). Checked: it answered 200 without a login for this open dataset.

So for Fairdata `mcls.acquire` uses one connection whatever `--connections` says (it prints `using 1, not N`), asks Etsin for a fresh token on every attempt, and restarts from byte 0 after a dropped transfer; there is no resume. The three files are 24.6, 55.0 and 72.1 GB. Metax v3 gives their checksums as sha256. Metax v2 labels the same 64-digit values SHA-384, but 64 hex digits is 256 bits, so sha256 is used.

## Measured speed per host

Measured 2026-11-12, 07:03 to 07:05 UTC, from the laptop, with curl and a generic User-Agent. Each request asked for one 10 MB (10,485,760-byte) byte range. "4 connections" means four different 10 MB ranges of the same file at once; the number is 40 MB divided by the time until the last one finished. Every response was `206`, and nothing was kept (all output went to `/dev/null`).

| Host (file used) | 1 connection | 4 connections |
| --- | --- | --- |
| Zenodo (`tampere/DJI_matrice_100_2G.bin`) | 0.47 MB/s (22.5 s) | 2.44 MB/s (17.2 s) |
| NIST, served from `nist-oar-cache.s3.amazonaws.com` (`nist_wifi_bt/2_4ghz_indoor.h5`) | 3.79 MB/s (2.8 s) | 5.49 MB/s (7.6 s) |
| Mendeley, served from S3 eu-west-1 (`dronerf/RF Data_10100_H.rar`) | 2.04 MB/s (5.1 s) | 2.74 MB/s (15.3 s) |
| Fairdata | not measured: it serves no ranges, and a stream test would start a 24 GB transfer | |

What this means:

- **Zenodo** limits each connection, as PLAN.md 3.7 says (about 0.6 MB/s before, 0.47 MB/s today). Four ranges are about five times faster. A 50 GB Zenodo file takes about 30 hours on one connection and about 6 hours on four.
- **NIST and Mendeley** are not limited per connection the same way. Four ranges add little, so one or two connections are enough.
- These are one-off numbers from one home line. Run `crc/check_env.sh` once on CRC before planning a large download.

On CRC, `prepare_array.job` runs up to 8 tasks at once with 1 connection each (`MCLS_CONNECTIONS=1`, `-tc 8`). That keeps about 8 connections on one host, the plan's limit. For Zenodo, the faster option is fewer tasks with more connections each, for example `-tc 2` with `MCLS_CONNECTIONS=4`.

## Refreshing the catalog

Re-read each dataset's `api` URL in the catalog and copy `name`, `size` and `checksum` from the response. Never type sizes or checksums by hand. For Mendeley, list each folder (`files?folder_id=<id>&version=1`), not `root`.
