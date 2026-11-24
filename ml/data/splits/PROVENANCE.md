# Official INCLUDE split files: provenance

These files are copied unmodified from the authors' code repository:

- Repo: https://github.com/AI4Bharat/INCLUDE (MIT license)
- Commit: `df348f4a640060d4d39f02556d9f96851e8c46c7`
- Paths: `train_test_paths/*.txt`, `label_maps/label_map_include.json`, `label_maps/label_map_include50.json`

Why vendored: the Zenodo record links the split as a Google Drive file
(`1tGjOR9xbk279ZUYnmk3BE-ktcWBdS24w`) that returned HTTP 404 on 2026-11-24, and
`Train_Test_Split.zip` (referenced by the record's own `download_data.sh`) is not in
the record. See `reports/phase0.md` and `docs/decisions/003-dataset-source-of-truth.md`.

The authors' `train` and `val` files together form the official training set (they sum to
the 3475 / 766 train counts stated on Zenodo). `test` is the official test set.

`ml/tests/test_splits.py` pins the sha256 of each file, so any edit fails CI.

Known quirks (handled in `sanket.data.splits`, never by editing these files):

- No trailing newline at end of file.
- 605 entries say `.MOV` where the Zenodo zip contains `.MP4`. Match by path stem.
- 8 entries belong to class `Days_and_Time/Second (Number)`, whose videos are not in any Zenodo zip.
