# FIFA Player Position Analysis

A Python data-analysis project exploring FIFA player data, with a focus on player positions and the attributes associated with different roles on the pitch.

## Overview

This repository contains the original Jupyter analysis together with the final project presentation. The workflow covers data cleaning, feature preparation, player-position grouping and exploratory analysis.

## Repository Contents

| File | Purpose |
| --- | --- |
| `project1_fifa_position_def.ipynb` | Original Python/Jupyter analysis |
| `FIFA PROJECT PRESENTATION 2023.pdf` | Final presentation |
| `data/README.md` | Instructions for restoring the source datasets locally |
| `requirements.txt` | Python dependencies |

## Data Setup

The notebook expects two source files:

- `FIFA_TRAIN_DATA.CSV`
- `FIFA_TEST_DATA.CSV`

Those datasets are **not currently included in the repository**. Their original source/licensing details are not documented well enough to redistribute them safely, so the project does not fabricate or silently replace them.

To run the analysis locally, place the two files in a `data/` directory:

```text
fifa-player-position-analysis/
├── data/
│   ├── FIFA_TRAIN_DATA.CSV
│   └── FIFA_TEST_DATA.CSV
├── project1_fifa_position_def.ipynb
└── README.md
```

Then use repository-relative paths in the notebook:

```python
from pathlib import Path

DATA_DIR = Path("data")
fifa_test = pd.read_csv(DATA_DIR / "FIFA_TEST_DATA.CSV", sep="?")
fifa_train = pd.read_csv(DATA_DIR / "FIFA_TRAIN_DATA.CSV", sep="?")
```

The historical notebook still contains the original local Mac paths from development; `data/README.md` documents the portable replacement.

## Analysis Focus

The project groups FIFA positions into broader categories such as defence, midfield, attack and goalkeeper, then prepares player attributes for analysis. The notebook includes cleaning steps for missing values, categorical fields, wages/values and physical attributes.

## Technologies

- Python
- Jupyter Notebook
- pandas / NumPy
- scikit-learn
- statsmodels
- matplotlib / seaborn

## Skills Demonstrated

- exploratory data analysis
- data cleaning and preparation
- feature engineering
- player-position categorization
- regression/model evaluation concepts
- sports-data analysis
- data visualization
- analytical storytelling

## How to Explore the Project

Start with the PDF presentation for a concise overview of the project, then review `project1_fifa_position_def.ipynb` for the detailed analytical workflow. To execute the notebook yourself, first follow the dataset instructions in `data/README.md`.
