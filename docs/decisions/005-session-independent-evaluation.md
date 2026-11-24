# ADR 005: Session-independent evaluation from camera-number blocks

- Status: **Partly decided.** The label and the spot-check result are accepted (Mukund,
  2026-11-24). The block-boundary rule below is **proposed**, to be finalised before Phase 3.
- Date: 2026-11-24

## Context

INCLUDE has no signer ids (`reports/phase0.md`). File names are camera counters
(`MVI_####`), and within a class consecutive takes have consecutive numbers. Mukund approved
holding out recording sessions, labelled **"session-independent", never "signer-independent"**,
on one condition: first spot-check about 10 camera-number runs to confirm that one run is one
signer/session. The authors have been asked for signer ids
(`docs/outreach/2026-11-email-include-authors.md`).

## Spot-check (done 2026-11-24)

Sample: `session_spotcheck` in `ml/data/samples/phase1_checks.json` (seed 0). It covers 5
classes × 2 runs = 10 runs, both cameras (.MOV and .MP4), and 5 categories (Electronics,
Days_and_Time, Society, Greetings, Jobs). For each run: the middle frame of the first take, the
last take, and the take numbered right after the run in a **different class** of the same
category (9 of the 10 runs have one). The sheets are built by `python -m sanket.checks.session_frames`
into the gitignored `ml/.cache/spotcheck/`. They are never committed, out of respect for the
signers (see the HF card's ethics note in ADR 003). The check was visual, done by Claude.
Mukund can re-run the command and look for himself.

| Hypothesis | Result |
|---|---|
| H1: a run (one class, consecutive numbers) is one signer in one setting | **10 / 10.** Same person, same clothes, same room and framing from first to last take |
| H2: the take right after a run, in another class, is the same sitting | **9 / 9.** Same person, clothes, room and framing |
| H3: two runs of the same class are different sessions | **10 / 10** pairs differ. In all 5 classes the two runs show different people |

Further observations:

- **A sitting spans many classes and categories.** One signer (the man in the red-checked
  shirt, door-side room) appears in the same clothes and room in Jobs (MVI_4782),
  Society (4845), Electronics (4967) and Days_and_Time (5060). In the manifest, camera numbers
  4770–5070 form one near-contiguous block across 9 categories.
- At least 6 distinct people appear in these 10 runs, across at least 2 rooms (whiteboard room,
  door-side room). That fits the 7 signers reported by secondary sources.
- The same person appears in different camera ranges (.MOV and .MP4). So holding out sessions
  does **not** hold out people. That is the reason for the label.

## Decision

1. The held-out unit is a **sitting**: a block of consecutive camera numbers **across all
   classes and categories**, not a per-class run. Holding out per-class runs would leave the
   same person, in the same clothes, on the same day in training under other words. That
   leakage is exactly what this evaluation exists to avoid.
2. The result is always called **session-independent**. If the authors provide signer ids, a
   real signer-independent split becomes a separate row. It never replaces this one.

## Proposed: how blocks are cut (to finalise before Phase 3)

A pure gap rule isn't safe by itself:

- With gap ≤ 3 per camera (extension) there are 26 blocks (largest 1,070 videos; 18 blocks of
  ≥ 50 videos cover 4,100 videos). But the 4770–5070 sitting has an internal gap (4811 → 4815),
  so a gap rule can **split one sitting**, which would leak.
- Because of counter wrap-around, 512 `.MOV` numbers are used twice. Sorting by number then
  **merges** unrelated cycles. That is safe (more conservative), but it makes blocks coarser.

Proposal: cut blocks with the gap rule, then **verify every block boundary visually** and merge
any boundary where the same person, clothes and room continue on both sides. That frames cost
nothing extra: during the full Phase 1 extraction every video is decoded anyway, so the extractor
also saves one small middle frame per video to the local, gitignored cache (about 130 MB, never
committed). The boundary review uses those frames, and the final block assignment is written to
a versioned file in `ml/data/splits/` with its own test.

## Tradeoff

- Visual verification is manual and only as good as the reviewer. It is limited to block
  boundaries (about 25 to 30 pairs), and each merge decision is recorded in the split file's
  provenance.
- Coarse blocks make the held-out set lumpy (one block can be about 25% of the data). The
  Phase 3 split picks whole blocks until roughly 20% of videos are held out, and reports the
  class coverage of the held-out set.
