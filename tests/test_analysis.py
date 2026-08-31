import math

import pandas as pd

from analysis import (
    clean_column_names,
    parse_height_cm,
    parse_money,
    parse_simple_expression,
    parse_stars,
    parse_weight_kg,
    position_group,
)


def test_clean_column_names():
    frame = pd.DataFrame(columns=["Player Name", " OVA "])
    cleaned = clean_column_names(frame)
    assert list(cleaned.columns) == ["player_name", "ova"]


def test_position_group_mapping():
    assert position_group("CB") == "defence"
    assert position_group("cam") == "midfield"
    assert position_group("ST") == "attack"
    assert position_group("GK") == "goalkeeper"
    assert position_group("unknown") == "unknown"


def test_parse_money():
    assert parse_money("€1.5M") == 1_500_000
    assert parse_money("€750K") == 750_000
    assert parse_money("€250") == 250


def test_parse_height_and_weight():
    assert math.isclose(parse_height_cm("6'0\""), 182.88, rel_tol=1e-6)
    assert math.isclose(parse_weight_kg("220lbs"), 99.7903214, rel_tol=1e-6)


def test_parse_stars():
    assert parse_stars("4 ★") == 4
    assert parse_stars("5") == 5


def test_safe_expression_parser():
    assert parse_simple_expression("75+2") == 77
    assert parse_simple_expression("81-1") == 80
    assert parse_simple_expression("hello") == "hello"
    # Arbitrary Python must not be evaluated.
    dangerous = "__import__('os').system('echo nope')"
    assert parse_simple_expression(dangerous) == dangerous
