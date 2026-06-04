# Silent Failure in PII Discovery — Saudi-Context Leakage Evaluation

Reproducibility package for the paper *"Silent Failure in PII Discovery:
Measuring Saudi-Context Leakage in Pattern- and Dictionary-Based Detection and
Its Implications for PDPL Compliance Assurance"* (IS566 Graduate Research
Project, King Saud University).

## Purpose

This repository contains the research artifacts needed to regenerate, from
scratch, the corpus and the results reported in the paper. The pipeline is
deterministic: a single fixed seed reproduces the exact dataset, and the scorer
derives every published number from it. The goal is to make the paper's leakage
measurements independently verifiable. The repository is intentionally limited
to the experiment itself — dataset generation, detector implementations,
scoring/evaluation, and the artifacts required to reproduce them.

> **All data is synthetic.** The corpus contains no real personal data. Names,
> National IDs, IBANs, phone numbers, e-mails, and National Addresses are
> programmatically generated; identifiers are format- and checksum-valid but
> correspond to no real person.

## Repository contents

| File | Role |
|------|------|
| `generate_dataset.py` | Builds the labelled corpus with exact character offsets |
| `detectors.py` | The four detector configurations and the seeded 70/30 name split used for the held-out evaluation |
| `score.py` | Entity-level scoring; computes P/R/F1, FNLR, SWLR, robustness gaps, in-dictionary vs. held-out recall, and Wilson 95% CIs |
| `make_figures.py` | Generates the paper's figures from `results.json` |
| `saudi_entities.py` | Curated name lists and synthetic identifier generators (National ID, IBAN, phone, e-mail, National Address) |
| `corpus.jsonl` | The generated corpus (regenerable; included for convenience) |
| `results.json` | The computed metrics (regenerable; included for convenience) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

`corpus.jsonl` and `results.json` are deterministic outputs of the pipeline and
are committed only for convenience; running the steps below regenerates them
identically. `make_figures.py` writes figure image files when run; those images
are build outputs and are not committed.

## Fixed seed

```
seed = 20260603
```

This is the project's submission date and is the default in
`generate_dataset.py`. The held-out name split in `detectors.py` uses its own
fixed split seed (also `20260603`); setting the train fraction to 1.0 reproduces
the full-coverage reference numbers reported in the paper.

## Requirements

- Python 3.10+
- `numpy`, `matplotlib`
- `presidio-analyzer` — **required to reproduce the published Presidio numbers.**
  If it is not installed, `detectors.py` degrades gracefully and the Presidio
  configuration simply returns no detections (the other three detectors are
  unaffected).

```bash
pip install -r requirements.txt
```

## Run order

```bash
# 1. Generate the labelled corpus (-> corpus.jsonl)
python3 generate_dataset.py --seed 20260603

# 2. Score every detector against the corpus (-> results.json)
python3 score.py

# 3. Generate the figures from the scored results
python3 make_figures.py
```

Each step depends on the previous one. `score.py` reads `corpus.jsonl` and
writes `results.json`; `make_figures.py` reads both `results.json` and
`corpus.jsonl`.

## Reproducing the results

Running the three commands above regenerates `corpus.jsonl` and `results.json`
identically to the committed versions, because the seed is fixed.
`generate_dataset.py` prints an offset-integrity check (`offset_mismatches=0`)
confirming every labelled span re-extracts to its stored surface form.
`score.py` prints the headline metrics, including the in-dictionary vs. held-out
recall breakdown.
