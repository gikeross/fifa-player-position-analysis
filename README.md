# FIFA Player Position Analysis

A Python sports-analytics project exploring how FIFA player attributes vary by position and how well position-specific regression models can explain a player's **Overall (OVA)** rating.

## Project Goal

The analysis groups detailed FIFA positions into four broader football roles:

- **Goalkeeper**
- **Defence**
- **Midfield**
- **Attack**

The aim is to clean player attributes, compare those groups, and fit separate regression models rather than assuming the same attributes matter equally for every position.

## Reproducible Portfolio Version

`analysis.py` is the recommended entry point. It:

1. loads either the historical project training split or the FIFA 21 raw dataset fallback from `data/`;
2. normalizes original FIFA column names such as `↓OVA` and `Best Position`;
3. groups detailed positions into four broad football roles;
4. converts value/wage, height, weight and star-rating fields into numeric features;
5. safely resolves simple FIFA attribute expressions such as `75+2` without Python `eval()`;
6. trains a separate linear-regression model for each position group;
7. evaluates each model using **MAE, RMSE and R²**;
8. exports model metrics, top coefficients and portfolio charts.

```bash
pip install -r requirements.txt
python analysis.py
```

## Dataset Provenance

The original notebook used project-specific files named `FIFA_TRAIN_DATA.CSV` and `FIFA_TEST_DATA.CSV`. Their exact Ironhack split could not be independently reconstructed.

However, the notebook's schema strongly matches the widely used **FIFA 21 raw data v2** dataset: **18,979 players and 77 columns**, including `OVA`, `POT`, `BOV`, `Best Position`, player attributes, financial fields and physical characteristics. Public descriptions of that dataset identify it as FIFA 21 / SoFIFA data distributed through Kaggle.

A currently available matching Kaggle copy is documented in [`data/README.md`](data/README.md). To use the reproducible fallback, place this file under `data/`:

```text
fifa21 raw data v2.csv
```

The script analyses that full dataset directly. It does **not** claim that this recreates the historical Ironhack train/test split.

If you still have `FIFA_TRAIN_DATA.CSV`, the script gives that historical training split priority instead.

## Expected Outputs

```text
outputs/
├── analysis_source.csv
├── position_distribution.csv
├── model_metrics.csv
└── top_model_features.csv

assets/
├── position_distribution.png
└── model_r2_by_position.png
```

`analysis_source.csv` explicitly records which input path produced the outputs, so results from the raw FIFA 21 fallback cannot be confused with the historical course split.

## Reliability and Code Quality

The historical notebook is preserved as the original exploratory work, while the portable implementation improves several areas:

- repository-relative paths instead of `/Users/...` paths;
- support for both historical and raw FIFA 21 schemas;
- reusable transformation functions instead of deeply nested `np.where` statements;
- safe expression parsing instead of dataframe-wide `eval()`;
- reproducible validation splits (`random_state=42`);
- MAE, RMSE and R² evaluation by position;
- automated transformation tests with `pytest`;
- GitHub Actions continuous integration.

The CI test suite currently passes successfully.

## Original Notebook

`project1_fifa_position_def.ipynb` contains the full historical cleaning and regression exploration. It remains in the repository to show the original development process, while `analysis.py` should be treated as the cleaner portfolio implementation.

## Repository Structure

```text
fifa-player-position-analysis/
├── analysis.py                         # Portable analysis/model pipeline
├── tests/test_analysis.py              # Transformation tests
├── .github/workflows/tests.yml         # Continuous integration
├── data/README.md                      # Dataset provenance/setup
├── project1_fifa_position_def.ipynb    # Original exploratory notebook
├── FIFA PROJECT PRESENTATION 2023.pdf  # Final project presentation
├── requirements.txt
└── README.md
```

## Technologies

- Python
- pandas / NumPy
- scikit-learn
- matplotlib
- statsmodels / Jupyter (historical notebook)
- pytest
- GitHub Actions

## Skills Demonstrated

- sports-data analysis
- data cleaning and feature engineering
- position-based segmentation
- regression modelling
- MAE / RMSE / R² evaluation
- data provenance and reproducibility
- safe parsing and transformation
- automated testing and CI

## Remaining Step

To publish concrete model results and charts, the matching FIFA 21 raw file still needs to be downloaded locally and placed in `data/`, or the original historical training file needs to be recovered. Once either is present, `python analysis.py` can generate the portfolio outputs without further code changes.
