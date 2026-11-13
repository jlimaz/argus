# Licence register: mod-classifier datasets

Every row comes from the source page named in it, opened on Nov 11-12, 2026. Labels and link decisions are in [configs/taxonomy.toml](configs/taxonomy.toml). Nothing here is legal advice. Where a page did not say something, the row says "not stated".

**Attribution:** every CC BY licence requires credit to the creator, a link to the licence, and a note of changes. NIST's terms ask users to "explicitly acknowledge the National Institute of Standards and Technology as the source of the data". So every dataset we use needs a credit line in the model card.

## Datasets in use

| Taxonomy name | Dataset | Licence (as the page states it) | Attribution required | Real? | Source opened |
| --- | --- | --- | --- | --- | --- |
| `tampere` | Radio-Frequency Control and Video Signal Recordings of Drones (Tampere University) | CC BY 4.0 (`cc-by-4.0` in the record) | Yes | Real (anechoic chamber) | https://zenodo.org/records/4264467, API https://zenodo.org/api/records/4264467 |
| `dronerf` | DroneRF dataset (Al-Sa'd et al., 2019) | CC BY 4.0. The data paper is also CC BY 4.0 | Yes. The authors ask that the Mendeley record be cited | Real (lab) | https://data.mendeley.com/datasets/f4c2b4n755/1; paper https://europepmc.org/article/PMC/PMC6727013; https://al-sad.github.io/DroneRF/ |
| `nist_wifi_bt` | Wi-Fi and Bluetooth I/Q recordings in the 2.4 and 5 GHz bands (NIST, mds2-2731) | NIST open licence (https://www.nist.gov/open/license). The README lists no other restriction ("N/A") | Yes (acknowledge NIST) | Real | https://catalog.data.gov/dataset/wi-fi-and-bluetooth-i-q-recordings-in-the-2-4-ghz-and-5-ghz-bands-with-low-cost-software-d; README https://data.nist.gov/od/ds/ark:/88434/mds2-2731/README.txt |
| `occupancy` | Multi-environment real-world multi-band RF dataset for spectrum sensing and occupancy (IEEE DataPort) | "Creative Commons Attribution" (**version not stated**) | Yes | Real | https://ieee-dataport.org/documents/multi-environment-real-world-multi-band-rf-dataset-spectrum-sensing-and-occupancy |
| `dronedetect` | DroneDetect dataset (IEEE DataPort, open access) | "Creative Commons Attribution" (**version not stated**) | Yes | Real | https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine |
| `fgi_jammer` | FGI Jammer Test 2023 Repository (Fairdata) | CC BY 4.0 | Yes | Real (live jamming test) | https://research.fi/en/results/dataset/2eee8cfd-db00-4e46-bd62-d7632c09c700 (Fairdata page https://etsin.fairdata.fi/dataset/2eee8cfd-db00-4e46-bd62-d7632c09c700 not opened) |
| `gateman` | GATEMAN project: GNSS and jammer raw data | CC BY 4.0 | Yes | Real (lab generators recorded by a USRP) | https://zenodo.org/records/2654322, API https://zenodo.org/api/records/2654322 |
| `triangular_l1` | GNSS L1 jammed signal, 10 MHz, triangular (part 1) | CC BY 4.0 | Yes | Real | https://zenodo.org/records/14859381, API https://zenodo.org/api/records/14859381 |
| `loraiq` | LoRaIQ: experimental LoRa dataset with annotated IQ samples (EPFL) | CC BY 4.0 | Yes | Real | https://zenodo.org/records/17708397, API https://zenodo.org/api/records/17708397 |
| `drexel_cbrs_radar` | CBRS band experimental waveform dataset for testing radar detection models (Drexel) | CC BY 4.0 | Yes | **Simulated waveforms** sent through a real USRP over cable (`is_real = false`) | https://zenodo.org/records/4521678, API https://zenodo.org/api/records/4521678 |
| `adsb` | 24-hour signal recording dataset with labels (ADS-B, IEEE DataPort) | "Creative Commons Attribution" (**version not stated**) | Yes | Real | https://ieee-dataport.org/open-access/24-hour-signal-recording-dataset-labels-cybersecurity-and-iot |
| `nist_radar` | RF dataset of incumbent radar systems in the 3.5 GHz CBRS band (NIST, mds2-2116) | NIST open licence (https://www.nist.gov/open/license). NIST employees' works "are not subject to copyright protection in the United States" but "may be subject to foreign copyright" | Yes (acknowledge NIST) | **Simulated** ("synthetically generated radar waveforms with added white Gaussian noise") | https://data.nist.gov/od/id/mds2-2116 (record via https://data.nist.gov/rmm/records?@id=ark:/88434/mds2-2116); terms https://www.nist.gov/open/license |

## Not usable yet

| Taxonomy name | Dataset | Status | Source opened |
| --- | --- | --- | --- |
| `rtl_433` | `rtl_433_tests` (sample `.cu8` captures) | **No licence found.** GitHub detects none, the repository root has no LICENSE file, and the README does not mention one. PLAN 3.1 says "GPL-2.0 (per QA log)", which is unconfirmed. Without a licence, default copyright applies. Excluded (`default_exclude`) until the maintainers state a licence. | https://github.com/merbanan/rtl_433_tests (API `repos/merbanan/rtl_433_tests`: `license: null`) |

## Excluded datasets

| Dataset | Reason for exclusion | What the opened pages say |
| --- | --- | --- |
| RFUAV (37 UAVs, about 1.3 TB) | PLAN 3.1 excludes it as CC BY-NC-SA (non-commercial). Also too large for the storage budget. | **Conflict.** The arXiv paper is CC BY-NC-SA 4.0 (https://arxiv.org/abs/2503.09033), but the code repository is Apache-2.0 (https://github.com/kitoweeknd/RFUAV) and the Hugging Face dataset card says `license: apache-2.0` (https://huggingface.co/datasets/alley66/RFUAV). The NC-SA label may belong to the paper only. **Open:** decide whether to reconsider. The 1.3 TB size still rules it out on the laptop. |
| Spectrum Highway, set 2 | PLAN 3.1: CC BY-NC-SA (non-commercial). | **Unconfirmed.** A web search found no page for a dataset by this name, so its licence could not be checked. The exclusion stands on PLAN's record only. |
| RadioML 2018.01A (DeepSig) | Non-commercial licence. Mostly synthetic channel effects. PLAN also cites reported label errors (not checked here). | DeepSig: "All datasets provided by Deepsig Inc. are licensed under the Creative Commons Attribution - NonCommercial - ShareAlike 4.0 License (CC BY-NC-SA 4.0)" (https://www.deepsig.ai/datasets/). |
| CardRF (Cardinal RF, University of Louisville) | PLAN 3.1: excluded "until its licence is recorded". | **Now recorded:** the IEEE DataPort page states "License: Creative Commons Attribution" (version not stated) (https://ieee-dataport.org/documents/cardinal-rf-cardrf-outdoor-uavuas-rf-signals-bluetooth-and-wifi-signals-dataset). The files themselves are served from a university SharePoint link whose terms were not opened. **Open:** the orchestrator decides whether to admit it. It has UAV, Bluetooth and Wi-Fi signals, over 65 GB, in .mat format. |
| WireTapper | PLAN 3.1: "not recommended in the QA log". | **Unconfirmed.** A web search found no dataset by this name. The exclusion stands on the QA log only. |

## Notes

- "Version not stated" rows (`occupancy`, `dronedetect`, `adsb`, CardRF): IEEE DataPort shows only "Creative Commons Attribution". Before publishing a model trained on them, confirm the version with the dataset authors or in the files' own licence text.
- `is_real = false` rows (`drexel_cbrs_radar`, `nist_radar`) may appear in training only. Every report prints metrics with and without them (PLAN 3.4).
