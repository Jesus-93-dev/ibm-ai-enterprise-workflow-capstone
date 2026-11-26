import json
import pandas as pd
from app.ingest import aggregate_daily, build_features, load_raw_transactions


def test_load_and_aggregate_daily(tmp_path):
    sample = [
        {"year": 2019, "month": 1, "day": 1, "country": "United Kingdom", "price": 100.0, "times_viewed": 5},
        {"year": 2019, "month": 1, "day": 1, "country": "France", "price": 50.0, "times_viewed": 2},
        {"year": 2019, "month": 1, "day": 3, "country": "United Kingdom", "price": 200.0, "times_viewed": 8},
    ]
    f = tmp_path / "batch_1.json"
    f.write_text(json.dumps(sample), encoding="utf-8")

    raw = load_raw_transactions(tmp_path)
    assert len(raw) == 3

    uk_daily = aggregate_daily(raw, country="United Kingdom")
    assert len(uk_daily) == 3  # rellena día faltante 2019-01-02
    assert uk_daily.loc[uk_daily["date"] == "2019-01-01", "revenue"].iloc[0] == 100.0
    assert uk_daily.loc[uk_daily["date"] == "2019-01-02", "revenue"].iloc[0] == 0.0

    feat = build_features(uk_daily)
    assert "lag_1" in feat.columns
    assert "target" in feat.columns
