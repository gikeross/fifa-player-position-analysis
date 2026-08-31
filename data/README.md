# FIFA Data Setup

The original project notebook used two project-specific files:

- `FIFA_TRAIN_DATA.CSV`
- `FIFA_TEST_DATA.CSV`

Those exact split files are not currently stored in this repository, and the exact Ironhack split/provenance could not be verified well enough to recreate them faithfully.

## Likely Underlying Dataset

The notebook schema strongly matches the widely used **FIFA 21 raw data v2** dataset: approximately **18,979 players and 77 columns**, with fields such as `OVA`, `POT`, `BOV`, `Best Position`, player attributes, value, wage, height, weight, skill-move ratings and work rates.

This dataset was scraped from SoFIFA and has been distributed through Kaggle under the filename:

```text
fifa21 raw data v2.csv
```

A currently available matching Kaggle copy is:

https://www.kaggle.com/datasets/abdelrahmanmohamed75/fifa21-dataset

The repository does **not** automatically redistribute that file. Download it from a source whose terms you accept and place it here as:

```text
data/fifa21 raw data v2.csv
```

Then run:

```bash
python analysis.py
```

`analysis.py` will detect the raw file, normalize its original column names (including `↓OVA` and `Best Position`) and generate the portfolio outputs directly from the full raw dataset.

## Historical Split Option

If you still have the original project files, place them here instead:

```text
fifa-player-position-analysis/
├── data/
│   ├── FIFA_TRAIN_DATA.CSV
│   └── FIFA_TEST_DATA.CSV
├── analysis.py
└── project1_fifa_position_def.ipynb
```

When `FIFA_TRAIN_DATA.CSV` is present, the portable analysis gives that historical split priority. The test file is optional for the current portfolio outputs because validation is performed by splitting the training data internally.

## Important Provenance Note

The full FIFA 21 raw dataset is a credible match for the source schema, but it is **not claimed to reproduce the exact historical train/test split** used in the original coursework. Results generated from `fifa21 raw data v2.csv` should therefore be described as a reproducible re-analysis of the same FIFA 21 schema rather than a reconstruction of the original competition split.

The historical notebook still contains the original local Mac paths used during development. `analysis.py` should be used for portable execution.
