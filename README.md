# FIFA Player Position Analysis

A Python sports-analytics project exploring how FIFA player attributes vary by position and how well position-specific regression models can explain a player's **Overall (OVA)** rating.

## Project Goal

The original analysis groups detailed FIFA positions into four broader football roles:

- **Goalkeeper**
- **Defence**
- **Midfield**
- **Attack**

The aim is to clean the raw player attributes, compare those groups, and fit separate regression models rather than assuming the same attributes matter equally for every position.

## Portfolio Version

`analysis.py` is now the recommended entry point for the project. It provides a portable and safer version of the core notebook workflow:

1. loads the training and test CSVs from `data/`;
2. standardizes column names;
3. groups detailed FIFA positions into broad football roles;
4. converts value/wage strings into numeric amounts;
5. converts height and weight into metric units;
6. extracts star ratings into numeric features;
7. safely resolves simple FIFA attribute expressions such as `75+2`;
8. trains a separate linear-regression model for each broad position group;
9. evaluates each model using **MAE, RMSE and R²**;
10. exports model metrics, top coefficients and portfolio charts.

Run it with:

```bash
pip install -r requirements.txt
python analysis.py
```

## Data Availability

The project expects:

```text
data/FIFA_TRAIN_DATA.CSV
data/FIFA_TEST_DATA.CSV
```

These datasets are **not currently included in the repository**. Their original source/licensing details were not documented well enough to redistribute them safely, so the repository does not fabricate or silently replace them.

See [`data/README.md`](data/README.md) for setup details.

Until the original files are restored, the automated tests validate the reusable cleaning/transformation logic, while full model outputs cannot be regenerated honestly.

## Expected Outputs

When the source CSVs are available, `analysis.py` generates:

```text
outputs/
├── position_distribution.csv
├── model_metrics.csv
└── top_model_features.csv

assets/
├── position_distribution.png
└── model_r2_by_position.png
```

The model metrics file reports the number of players/features plus validation **MAE, RMSE and R²** for each position group. `top_model_features.csv` records the largest absolute linear-regression coefficients for each model.

## Reliability and Code Quality

The historical notebook is preserved as the original exploratory work, but the portable script corrects several maintainability issues:

- repository-relative paths instead of `/Users/...` paths;
- reusable transformation functions rather than long nested `np.where` statements;
- safe parsing of numeric expressions instead of applying Python `eval()` across the dataframe;
- explicit missing-data checks;
- reproducible train/validation splits (`random_state=42`);
- automated transformation tests with `pytest`;
- GitHub Actions CI on relevant pushes and pull requests.

Run the tests locally with:

```bash
pytest -q
```

## Original Notebook

`project1_fifa_position_def.ipynb` contains the full historical analysis, including the original data-cleaning workflow, position categorization and regression exploration. It is retained to show the development process, while `analysis.py` should be treated as the cleaner portfolio implementation.

## Repository Structure

```text
fifa-player-position-analysis/
├── analysis.py                         # Portable analysis/model pipeline
├── tests/test_analysis.py              # Transformation tests
├── .github/workflows/tests.yml         # Continuous integration
├── data/README.md                      # Dataset restoration instructions
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
- categorical domain mapping
- reproducible Python workflows
- safe parsing and data transformation
- automated testing and CI

## Current Limitation

The largest remaining gap is data provenance. Once the original FIFA CSV source is identified and redistribution rights are clear, the best next step is to restore the data (or provide a documented download link), run `analysis.py`, commit the generated metrics/charts, and surface the strongest model findings directly in this README.
