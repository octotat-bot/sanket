# Sanket

Real-time Indian Sign Language (ISL) **word** recognition that runs entirely in the browser.

> Status: **Phase 0 (setup and verification)**. Nothing to demo yet. Progress is tracked in
> [PROGRESS.md](PROGRESS.md) and findings in [reports/phase0.md](reports/phase0.md).

This is isolated word-level sign recognition for a limited vocabulary. It is not a sign-language
translator and not a replacement for interpreters. ISL is a complete language with its own
grammar.

## Run locally

Requires [uv](https://docs.astral.sh/uv/), Node 22, and pnpm 12.

```bash
make setup     # pinned Python + web environments
make ci        # lint, types, tests, MediaPipe smoke tests (Python + Chromium), build
```

## Dataset credit

Trained on **INCLUDE** (Sridhar, Ganesan, Kumar, Khapra; AI4Bharat / IIT Madras; ACM MM 2020,
[doi:10.1145/3394171.3413528](https://doi.org/10.1145/3394171.3413528)), hosted on
[Zenodo](https://zenodo.org/records/4010759). Official split files are from
[AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) (MIT).
License wording is pending confirmation (see `reports/phase0.md`, discrepancy 1). This project
is non-commercial.
