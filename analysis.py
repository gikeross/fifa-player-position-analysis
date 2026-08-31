"""Portable FIFA position analysis pipeline.

The original project notebook is preserved for historical context. This script provides
an executable, safer version of the core workflow using repository-relative data paths.

Preferred historical inputs (not redistributed in this repository):
    data/FIFA_TRAIN_DATA.CSV
    data/FIFA_TEST_DATA.CSV

Reproducible fallback input:
    data/fifa21 raw data v2.csv

The fallback file matches the 18,979-row / 77-column FIFA 21 raw dataset schema used by
the original project. It is analysed directly and is NOT presented as a reconstruction
of the original Ironhack train/test split.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
ASSET_DIR = ROOT / "assets"
TRAIN_PATH = DATA_DIR / "FIFA_TRAIN_DATA.CSV"
TEST_PATH = DATA_DIR / "FIFA_TEST_DATA.CSV"
RAW_PATH = DATA_DIR / "fifa21 raw data v2.csv"

POSITION_GROUPS = {
    "CB": "defence",
    "RWB": "defence",
    "LB": "defence",
    "LWB": "defence",
    "RB": "defence",
    "CM": "midfield",
    "CAM": "midfield",
    "CDM": "midfield",
    "RM": "midfield",
    "LM": "midfield",
    "CF": "attack",
    "ST": "attack",
    "RW": "attack",
    "LW": "attack",
    "GK": "goalkeeper",
}

COLUMN_ALIASES = {
    "↓ova": "ova",
    "best_position": "bp",
    "preferred_foot": "foot",
    "long_name": "longname",
    "photo_url": "player_photo",
    "flag_photo_url": "flag_photo",
    "club_logo_url": "club_logo",
}

DROP_COLUMNS = {
    "unnamed:_0",
    "nationality",
    "club",
    "bp",
    "position",
    "positions",
    "player_photo",
    "photourl",
    "playerurl",
    "club_logo",
    "flag_photo",
    "team_&_contract",
    "foot",
    "joined",
    "release_clause",
    "loan_date_end",
    "contract",
    "gender",
    "a/w",
    "d/w",
    "name",
    "longname",
    "id",
    "bov",
    "pot",
}

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
}


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    cleaned = [
        str(col).strip().lower().replace(" ", "_").replace("-", "_")
        for col in frame.columns
    ]
    frame.columns = [COLUMN_ALIASES.get(col, col) for col in cleaned]
    return frame


def position_group(position: object) -> str:
    if pd.isna(position):
        return "unknown"
    return POSITION_GROUPS.get(str(position).strip().upper(), "unknown")


def parse_money(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace("€", "").replace(",", "")
    multiplier = 1.0
    if text.endswith("M"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("K"):
        multiplier = 1_000.0
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def parse_height_cm(value: object) -> float:
    """Convert FIFA height strings such as 5'11\" to centimetres."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace('"', "")
    if "'" not in text:
        numeric = pd.to_numeric(text.replace("cm", "").strip(), errors="coerce")
        return float(numeric) if pd.notna(numeric) else np.nan

    try:
        feet_text, inches_text = text.split("'", 1)
        total_inches = int(feet_text) * 12 + int(inches_text)
        return total_inches * 2.54
    except (TypeError, ValueError):
        return np.nan


def parse_weight_kg(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).lower().strip()
    if text.endswith("kg"):
        number = pd.to_numeric(text.removesuffix("kg").strip(), errors="coerce")
        return float(number) if pd.notna(number) else np.nan

    text = text.replace("lbs", "").replace("lb", "").strip()
    number = pd.to_numeric(text, errors="coerce")
    if pd.isna(number):
        return np.nan
    return float(number) * 0.45359237


def parse_stars(value: object) -> float:
    if pd.isna(value):
        return np.nan
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return float(digits) if digits else np.nan


def parse_simple_expression(value: object) -> object:
    """Safely evaluate simple FIFA rating expressions such as '75+2' or '81-1'.

    The historical notebook used Python ``eval`` across the entire dataframe. This
    parser intentionally supports only numeric literals plus addition/subtraction.
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text or not any(sign in text for sign in ("+", "-")):
        return value

    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return value

    def _evaluate(item: ast.AST) -> float:
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)):
            return float(item.value)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub):
            return -_evaluate(item.operand)
        if isinstance(item, ast.BinOp) and type(item.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(item.op)](_evaluate(item.left), _evaluate(item.right))
        raise ValueError("Unsupported expression")

    try:
        return _evaluate(node)
    except (TypeError, ValueError):
        return value


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = clean_column_names(frame)

    if "bp" in frame.columns:
        frame["position_group"] = frame["bp"].map(position_group)

    for column in ("value", "wage"):
        if column in frame.columns:
            frame[column] = frame[column].map(parse_money)

    if "height" in frame.columns:
        frame["height"] = frame["height"].map(parse_height_cm)
    if "weight" in frame.columns:
        frame["weight"] = frame["weight"].map(parse_weight_kg)

    for column in ("sm", "w/f", "ir"):
        if column in frame.columns:
            frame[column] = frame[column].map(parse_stars)

    for column in frame.select_dtypes(include="object").columns:
        frame[column] = frame[column].map(parse_simple_expression)

    return frame


def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """Load the historical split when available, otherwise use the verified raw schema."""
    if TRAIN_PATH.exists():
        train = prepare_frame(pd.read_csv(TRAIN_PATH, sep="?"))
        test = prepare_frame(pd.read_csv(TEST_PATH, sep="?")) if TEST_PATH.exists() else None
        return train, test, "historical-train-split"

    if RAW_PATH.exists():
        raw = prepare_frame(pd.read_csv(RAW_PATH, low_memory=False))
        return raw, None, "fifa21-raw-data-v2"

    raise FileNotFoundError(
        "No FIFA source dataset found. Add either FIFA_TRAIN_DATA.CSV (and optionally "
        "FIFA_TEST_DATA.CSV) or fifa21 raw data v2.csv under data/. See data/README.md."
    )


def numeric_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns], errors="ignore")
    candidates = candidates.drop(columns=["ova", "position_group"], errors="ignore")
    return candidates.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")


def fit_position_models(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "ova" not in train.columns:
        raise ValueError("Dataset must contain the target column 'ova'.")
    if "position_group" not in train.columns:
        raise ValueError("Dataset must contain a usable best-position column ('bp' / 'Best Position').")

    metric_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for group in ("goalkeeper", "defence", "midfield", "attack"):
        subset = train.loc[train["position_group"] == group].copy()
        if len(subset) < 30:
            continue

        y = pd.to_numeric(subset["ova"], errors="coerce")
        X = numeric_feature_frame(subset)
        valid = y.notna()
        X = X.loc[valid]
        y = y.loc[valid]

        X = X.loc[:, X.notna().mean() >= 0.7]
        if len(X) < 30 or X.shape[1] == 0:
            continue

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", LinearRegression()),
            ]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_valid)

        metric_rows.append(
            {
                "position_group": group,
                "players": len(subset),
                "features": X.shape[1],
                "mae": mean_absolute_error(y_valid, predictions),
                "rmse": mean_squared_error(y_valid, predictions) ** 0.5,
                "r2": r2_score(y_valid, predictions),
            }
        )

        coefficients = model.named_steps["regressor"].coef_
        ranked = sorted(zip(X.columns, coefficients), key=lambda pair: abs(pair[1]), reverse=True)[:10]
        for feature, coefficient in ranked:
            feature_rows.append(
                {
                    "position_group": group,
                    "feature": feature,
                    "coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(feature_rows)


def generate_outputs(train: pd.DataFrame, source_label: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ASSET_DIR.mkdir(exist_ok=True)

    pd.DataFrame([{"analysis_source": source_label, "rows": len(train)}]).to_csv(
        OUTPUT_DIR / "analysis_source.csv", index=False
    )

    distribution = (
        train["position_group"]
        .value_counts(dropna=False)
        .rename_axis("position_group")
        .reset_index(name="players")
    )
    distribution.to_csv(OUTPUT_DIR / "position_distribution.csv", index=False)

    plot_distribution = distribution.loc[distribution["position_group"] != "unknown"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(plot_distribution["position_group"], plot_distribution["players"])
    ax.set_title("FIFA players by broad position group")
    ax.set_xlabel("Position group")
    ax.set_ylabel("Players")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "position_distribution.png", dpi=160)
    plt.close(fig)

    metrics, features = fit_position_models(train)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    features.to_csv(OUTPUT_DIR / "top_model_features.csv", index=False)

    if not metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(metrics["position_group"], metrics["r2"])
        ax.set_title("Position-specific linear regression validation R²")
        ax.set_xlabel("Position group")
        ax.set_ylabel("R²")
        fig.tight_layout()
        fig.savefig(ASSET_DIR / "model_r2_by_position.png", dpi=160)
        plt.close(fig)


def main() -> None:
    train, _test, source_label = load_data()
    generate_outputs(train, source_label)
    print(f"Generated FIFA portfolio outputs from {source_label} in outputs/ and assets/.")


if __name__ == "__main__":
    main()
