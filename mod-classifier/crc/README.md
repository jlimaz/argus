# mod-classifier on CRC

Shell scripts to set up Notre Dame's CRC, prepare the datasets and train. They were used for waves 1 to 5 from Odon's account (Nov 12-13, 2026); the log of the last wave is in [../docs/history/OVERNIGHT_PLAN.md](../docs/history/OVERNIGHT_PLAN.md). Each job file's header comment is the reference for its arguments and settings.

| File | What it does |
| --- | --- |
| `common.sh` | Helpers the other scripts source: find and load a `python/3.12` module, activate the venv |
| `setup_env.sh` | Front end: load Python 3.12, create the venv, install `../requirements.txt`, print torch and CUDA status |
| `check_env.sh` | Front end, read-only: quota, groups, free space, modules, free GPUs, Zenodo download speed |
| `check_node.sh`, `check_node.job` | Compute node, read-only: `/tmp` size, CPUs, RAM, `nvidia-smi`, torch sees CUDA |
| `prepare_array.job` | Array job: one raw file per task, fetch to node `/tmp`, preprocess, keep outputs, delete raw |
| `prepare_rtl433.job` | Prepares rtl_433_tests (`other_ism`), which the catalog cannot fetch: one clone step, then an array over device folders |
| `build_manifest.job` | Merges prepared datasets into the shards folder and builds one manifest for a class set |
| `train.job` | Trains one model from a config, then calibrates and evaluates it |
| `submit_wave.sh` | Submits a whole wave as a chain: manifest job, one training job per model, summary job |
| `wave_summary.job` | Emails the results table, plots and logs when a wave ends |
| `CI_PROPOSAL.md` | Draft: how the repo CI could also run `mod-classifier/tests` (needs Gabriel's agreement) |
| `../scripts/make_file_list.py` | Writes `runs/file_lists/files_<dataset>.txt` from `configs/catalog.toml`; prints the array size |
| `../scripts/space_needed.py` | Bytes of node `/tmp` one task needs (file, plus extracted size when unpacking); the job checks it before downloading |
| `../scripts/merge_shards.py` | Gathers the finished per-file outputs into one shards dir for `build_manifest` |

## Settings

All scripts read these environment variables (all optional):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MCLS_VENV` | `$HOME/venvs/mcls` | Where the venv lives. Put it on `/groups/<PI netid>` if that exists and home is tight |
| `MCLS_PYTHON_MODULE` | newest `python/3.12*` | Module to load. `setup_env.sh` saves its choice in the venv, so jobs reuse it |
| `MCLS_TORCH_INDEX_URL` | none | Extra pip index for a specific torch CUDA build, e.g. `https://download.pytorch.org/whl/cu121` |

## Order to run

Log in to a front end (`ssh <netid>@crcfe01.crc.nd.edu`), clone the repo (`main`), then:

```bash
cd argus/mod-classifier
mkdir -p runs/crc_logs                                  # job logs land here (git-ignored)

# 1. What the account offers (about a minute, mostly the speed test)
bash crc/check_env.sh 2>&1 | tee runs/check_env.txt

# 2. Build the venv (several minutes; torch with CUDA is a few GB)
bash crc/setup_env.sh 2>&1 | tee runs/setup_env.txt

# 3. One GPU node and one CPU node (each a minute or less once it starts)
qsub -q gpu -l gpu_card=1 crc/check_node.job
qsub -q long crc/check_node.job
qstat -u $USER                                          # wait until both are gone
cat runs/crc_logs/check_node.o*
```

Run step 1 before step 2: if home is nearly full, set `MCLS_VENV` to a path on group storage first.

Step 4 prepares one dataset. It needs `mcls.acquire` (Engineer 2) and `mcls.preprocess` (Engineer 4) on the branch, and Odon's approval for that dataset's download (file names, source and total size first: `python -m mcls.acquire --dataset tampere --list`). If a module is missing, or `mcls.sources.get(<dataset>)` fails (no loader, or no taxonomy entry for it), the job stops at once with exit code 3, before it downloads anything. It also stops with exit code 3 if node `/tmp` (by `df`) has less free space than the file needs (see `MCLS_UNPACK` below).

```bash
# 4a. File list from the catalog; the number printed is the array size
N=$(python scripts/make_file_list.py tampere)           # -> runs/file_lists/files_tampere.txt

# 4b. One task per file (at most 8 at once, 1 connection each)
qsub -t 1-$N -v MCLS_DATASET=tampere,MCLS_FILE_LIST=$PWD/runs/file_lists/files_tampere.txt crc/prepare_array.job
qstat -u $USER                                          # wait until all tasks are gone

# 4c. Merge the finished files; lists any file without output
python scripts/merge_shards.py --prepared ~/mcls_data/prepared/tampere \
    --file-list runs/file_lists/files_tampere.txt --dest ~/mcls_data/shards
#     missing files? rerun 4b unchanged: tasks whose output has DONE are skipped

# 4d. Manifest and splits (Engineer 5), over every merged dataset
python -m mcls.dataset.build_manifest --shards ~/mcls_data/shards --out data/manifest.jsonl --seed 1
```

Repeat 4a to 4c per dataset into the same `--dest`, then run 4d once.

## What each prints and how to read it

**`check_env.sh`**: one `===== heading =====` per check. A missing command prints `(command not found: ...)` and the script carries on.

- *quota*: used and limit for home (100 GB, fixed). If a `/groups/<name>` directory exists for one of your groups, it is listed with its quota and `df`. "none of your groups has a /groups/<name> directory" means no faculty-sponsored group storage, so everything (venv, prepared windows) must fit in 100 GB.
- *free space of $HOME*: the filesystem's free space; the quota above is the real limit.
- *module avail python / cuda / pytorch*: what the cluster provides. The last line says which `python/3.12` module `setup_env.sh` will pick; "none found" means setup will fail until a module is named with `MCLS_PYTHON_MODULE`.
- *free_gpus.sh @crc_gpu*: free GPU cards per node right now.
- *Zenodo speed test*: `RESULT 1 connection` and `RESULT 4 connections` in MB/s. From the laptop, one connection gave about 0.6 MB/s. If four connections give about four times one, the cap is per connection, as at home, and jobs should use 4 to 8 connections. Each part should say `http 206`. Any other code means the range request was refused.

**`setup_env.sh`**: ends with `torch <version> built for CUDA <version>` and `cuda available: False` on a front end (no GPU there; that is expected). Rerunning prints "exists, reusing" and "requirements.txt unchanged ..., skipping pip". `venv size` shows how much of the quota the venv uses.

**`check_node.job`** (log `runs/crc_logs/check_node.o<job id>`):

- *df -h /tmp*: node-local scratch. Its free space caps the largest raw file a task can fetch (NIST files are about 20 GB, some archives about 50 GB).
- *CPUs*: `nproc` is what the job may use, `nproc --all` the whole node.
- *RAM*: `free -g`.
- *nvidia-smi*: the GPU model, memory, and driver/CUDA version (GPU node only; "command not found" on a CPU node is normal).
- *torch inside the venv*: on the GPU node, `cuda available: True`, the device name and `matmul on GPU ok: True` mean training can run. `False` on the GPU node means the torch build and the driver do not match: rerun `setup_env.sh` with `MCLS_TORCH_INDEX_URL` set to the CUDA version `nvidia-smi` reports (`MCLS_FORCE_INSTALL=1`).

## `prepare_array.job` (template)

One task per raw file. Task N reads line N of a file list (blank lines and `#` comments skipped). The names are the catalog's `name` fields, written by `scripts/make_file_list.py`; it refuses a dataset that `mcls.acquire` cannot fetch unattended (`ieee_login`, `manual`) or whose file list is unknown in the catalog. It lists only recordings: files whose catalog `role` is `data` (no role means data). Files marked `metadata` (label tables, calibration) or `doc` (readers, scripts, figures) are left out, each named on stderr, since preprocess would find no recordings in them; a file with no role but a documentation suffix (`.py`, `.m`, `.pdf`, ...) is left out as a backstop. `--keep-all` lists everything. `--match TEXT` keeps a subset, for example one NIST radar group. LoRaIQ and Drexel need no `--match`: their catalog roles leave one data file each. Each task then:

1. `python -m mcls.acquire --dataset D --only NAME --connections N`, which writes `$MCLS_RAW/<base name of NAME>` with `MCLS_RAW=/tmp/$USER/mcls_raw/<job>.<task>`
2. `python -m mcls.preprocess --dataset D --raw $MCLS_RAW --out <node scratch>`
3. copies only the outputs to `$MCLS_OUT/<key>/` and writes a `DONE` marker there; `<key>` is the file's base name, which `mcls.acquire` keeps unique per dataset
4. deletes the raw file and the scratch, also when the task fails

`merge_shards.py` then hard-links each finished `<key>/` folder, unchanged, into `<dest>/<dataset>/<key>/`, so the relative shard paths in its `.jsonl` rows stay valid. It skips the hidden `.<key>.partial.*` folders of killed tasks, which `build_manifest`'s recursive search would otherwise read. `--move` renames instead of linking.

`MCLS_OUT` defaults to `$HOME/mcls_data/prepared/<dataset>`; `MCLS_CONNECTIONS` defaults to 1. `#$ -tc 8` lets at most 8 tasks run (and download from Zenodo) at once, so about 8 connections in total, the cap in PLAN.md. Keep tasks x connections near 8: if you raise `MCLS_CONNECTIONS`, lower `-tc` (for example `MCLS_CONNECTIONS=4` with `qsub -tc 2`). The job refuses `MCLS_CONNECTIONS` above 8, and `mcls.acquire` also applies each host's own cap (1 for Fairdata). Set `MCLS_MAX_GB` to make a task refuse a file larger than the node's free `/tmp`. Set `MCLS_UNPACK=1` for a dataset whose loader reads extracted files only (GATEMAN: its loader asks for `rawdata.zip` to be extracted first): the task unpacks the `.zip` or `.tar` with Python's own modules and deletes the archive before preprocessing, so the node needs room for the archive and its contents at once. Before downloading, the task compares that total with `df` of node `/tmp` and stops (exit 3) if it does not fit. The extracted size comes from the archive's `unpacked_size` in the catalog; GATEMAN's entry has none yet, so until it does, pass it by hand with `MCLS_UNPACKED_GB=<GB>`, or the task refuses rather than guess. A rerun skips files whose output has a `DONE` marker. Set `MCLS_KEEP_FAILED=1` to keep a failed task's files in `/tmp` for debugging.

## Trained weights: never in git

A checkpoint is about 4 MB (PLAN.md section 10). Weights never go into the git history, not even small ones: once committed they stay in every clone for good. Put each released checkpoint in a GitHub release asset or in the team's Drive, and record its SHA-256 in the model card next to the run folder and manifest hash it came from:

```bash
shasum -a 256 runs/<name>/model.pt       # macOS; on Linux: sha256sum runs/<name>/model.pt
```

`runs/` is git-ignored, so checkpoints written there are not committed by accident. Which of the two stores to use is still for the team to agree.

## What to send back to the orchestrator

Paste these in full. They hold no secrets, but replace your NetID with `<netid>` wherever it appears:

1. `runs/check_env.txt`: especially the quota lines, the `/groups` section, the module lists, and both `RESULT` speed lines.
2. The last 15 lines of `runs/setup_env.txt` (torch version, CUDA build, venv size), or the full error if it failed.
3. Both `runs/crc_logs/check_node.o*` logs, with the queue each came from (gpu or long), and how long each waited in `qstat` before starting.
4. Anything that did not work as this README says.
