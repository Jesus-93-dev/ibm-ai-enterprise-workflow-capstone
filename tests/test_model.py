import numpy as np
import pandas as pd
from app.model import load_artifact, predict_revenue, train_model


def _make_synthetic_df(days: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=days, freq="D")
    rows = []
    for idx, d in enumerate(dates):
        for country, mult in [("United Kingdom", 1.0), ("Germany", 0.6)]:
            rows.append(
                {
                    "year": d.year,
                    "month": d.month,
                    "day": d.day,
                    "country": country,
                    "price": float(120.0 * mult + 15.0 * np.sin(idx / 5.0)),
                    "times_viewed": int(10 * mult),
                }
            )
    return pd.DataFrame(rows)


def test_train_and_predict_isolated(tmp_path):
    raw = _make_synthetic_df(80)
    res = train_model(raw, model_dir=tmp_path / "models", country="all")
    assert res["rows_daily"] == 80

    art = load_artifact(tmp_path / "models", country="all")
    pred = predict_revenue(art, target_date="2019-04-01", country="all")
    assert pred["predicted_revenue"] >= 0.0
    assert "novelty_detected" in pred
