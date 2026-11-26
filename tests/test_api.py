import json
import numpy as np
import pandas as pd
from app.api import create_app


def _write_json_dataset(target_dir):
    dates = pd.date_range("2019-01-01", periods=80, freq="D")
    records = []
    for idx, d in enumerate(dates):
        for country, base in [("United Kingdom", 150.0), ("Germany", 90.0)]:
            records.append(
                {
                    "year": int(d.year),
                    "month": int(d.month),
                    "day": int(d.day),
                    "country": country,
                    "price": float(base + 10.0 * np.cos(idx / 4.0)),
                    "times_viewed": 12,
                }
            )
    (target_dir / "data.json").write_text(json.dumps(records), encoding="utf-8")


def test_api_endpoints_isolated(tmp_path):
    data_dir = tmp_path / "cs-train"
    data_dir.mkdir()
    _write_json_dataset(data_dir)

    app = create_app(
        {
            "TESTING": True,
            "DATA_DIR": str(data_dir),
            "MODEL_DIR": str(tmp_path / "models"),
            "LOG_PATH": str(tmp_path / "logs" / "test.jsonl"),
        }
    )
    client = app.test_client()

    r_health = client.get("/health")
    assert r_health.status_code == 200

    # Entrenar global y por país
    assert client.post("/train", json={"country": "all"}).status_code == 200
    assert client.post("/train", json={"country": "United Kingdom"}).status_code == 200

    # Predicción global y por país específico
    r_pred_all = client.post("/predict", json={"target_date": "2019-05-01", "country": "all"})
    assert r_pred_all.status_code == 200
    assert "predicted_revenue" in r_pred_all.get_json()

    r_pred_uk = client.post("/predict", json={"target_date": "2019-05-01", "country": "United Kingdom"})
    assert r_pred_uk.status_code == 200
    assert r_pred_uk.get_json()["country"] == "United Kingdom"

    # Error por entrada faltante
    r_bad = client.post("/predict", json={"country": "all"})
    assert r_bad.status_code == 400

    # Consultar logs
    r_logs = client.get("/logs")
    assert r_logs.status_code == 200
    assert r_logs.get_json()["count"] >= 2
