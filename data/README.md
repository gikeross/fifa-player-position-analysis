# FIFA Data Setup

The original project notebook expects two source datasets:

- `FIFA_TRAIN_DATA.CSV`
- `FIFA_TEST_DATA.CSV`

These files are not currently stored in this repository. To run the analysis locally, place both files in this `data/` directory.

The portable project path convention is therefore:

```text
fifa-player-position-analysis/
├── data/
│   ├── FIFA_TRAIN_DATA.CSV
│   └── FIFA_TEST_DATA.CSV
├── project1_fifa_position_def.ipynb
└── README.md
```

The historical notebook still contains the original local Mac paths used during development. When running it, replace those two `pd.read_csv(...)` calls with:

```python
from pathlib import Path

DATA_DIR = Path("data")
fifa_test = pd.read_csv(DATA_DIR / "FIFA_TEST_DATA.CSV", sep="?")
fifa_train = pd.read_csv(DATA_DIR / "FIFA_TRAIN_DATA.CSV", sep="?")
```

The source datasets are intentionally not fabricated or redistributed here because their original licensing/source details are not documented in the repository.
