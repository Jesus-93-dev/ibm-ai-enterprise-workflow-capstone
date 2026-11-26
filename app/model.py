import pickle
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from app.ingest import aggregate_daily, build_features


FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "roll_mean_7",
    "roll_mean_30",
    "views_lag_1",
    "purchases_lag_1",
    "month",
    "day_of_week",
]


def train_model(
    raw_df: pd.DataFrame,
    model_dir: str | Path = "models",
    country: str = "all",
) -> dict[str, Any]:
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = aggregate_daily(raw_df, country=country)
    feat_df = build_features(daily)
    usable = feat_df.dropna(subset=FEATURE_COLS + ["target"]).copy()

    if len(usable) >= 20:
        split_idx = int(len(usable) * 0.8)
        train_part = usable.iloc[:split_idx]
        val_part = usable.iloc[split_idx:]

        X_train, y_train = train_part[FEATURE_COLS], train_part["target"]
        X_val, y_val = val_part[FEATURE_COLS], val_part["target"]

        model = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08, max_iter=120, random_state=42)
        model.fit(X_train, y_train)

        val_pred = np.clip(model.predict(X_val), 0.0, None)
        base_pred = np.clip(val_part["roll_mean_7"].fillna(y_train.mean()).to_numpy(), 0.0, None)

        rmse_model = float(root_mean_squared_error(y_val, val_pred))
        mae_model = float(mean_absolute_error(y_val, val_pred))
        rmse_base = float(root_mean_squared_error(y_val, base_pred))
        mae_base = float(mean_absolute_error(y_val, base_pred))
        model_type = "HistGradientBoostingRegressor"
    else:
        fallback_val = float(daily["revenue"].mean()) if not daily.empty else 0.0
        model = Ridge(alpha=1.0)
        if len(daily) >= 2:
            x_idx = np.arange(len(daily)).reshape(-1, 1)
            model.fit(x_idx, daily["revenue"].to_numpy())
        rmse_model = 0.0
        mae_model = 0.0
        rmse_base = 0.0
        mae_base = 0.0
        model_type = f"RidgeFallback(mean={fallback_val:.2f})"

    artifact = {
        "model": model,
        "model_type": model_type,
        "country": country,
        "feature_cols": FEATURE_COLS,
        "daily_history": daily,
        "revenue_mean": float(daily["revenue"].mean()) if not daily.empty else 0.0,
        "revenue_std": float(daily["revenue"].std()) if len(daily) > 1 else 1.0,
        "metrics": {
            "model_rmse": rmse_model,
            "model_mae": mae_model,
            "baseline_rmse": rmse_base,
            "baseline_mae": mae_base,
        },
    }

    safe_country = country.lower().replace(" ", "_")
    artifact_path = out_dir / f"model_{safe_country}.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    return {
        "country": country,
        "model_type": model_type,
        "artifact_path": str(artifact_path),
        "rows_daily": int(len(daily)),
        "metrics": artifact["metrics"],
    }


def load_artifact(model_dir: str | Path, country: str = "all") -> dict[str, Any]:
    safe_country = country.lower().replace(" ", "_")
    artifact_path = Path(model_dir) / f"model_{safe_country}.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No existe modelo entrenado para '{country}' en {artifact_path}")
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def predict_revenue(
    artifact: dict[str, Any],
    target_date: str,
    country: str = "all",
) -> dict[str, Any]:
    parsed_date = pd.to_datetime(target_date, errors="raise").normalize()
    daily = artifact.get("daily_history", pd.DataFrame())
    feat_df = build_features(daily) if not daily.empty else pd.DataFrame()

    novelty = False
    mean_rev = float(artifact.get("revenue_mean", 0.0))
    std_rev = max(float(artifact.get("revenue_std", 1.0)), 1.0)

    if not feat_df.empty and hasattr(artifact.get("model"), "predict"):
        last_row = feat_df.iloc[[-1]].copy()
        last_row["month"] = parsed_date.month
        last_row["day_of_week"] = parsed_date.dayofweek
        if not last_row[FEATURE_COLS].isna().any().any():
            pred_val = float(artifact["model"].predict(last_row[FEATURE_COLS])[0])
        else:
            pred_val = float(daily["revenue"].tail(7).mean())
    else:
        pred_val = mean_rev

    pred_val = max(0.0, pred_val)
    if abs(pred_val - mean_rev) > 3.0 * std_rev:
        novelty = True

    return {
        "country": country,
        "target_date": parsed_date.strftime("%Y-%m-%d"),
        "predicted_revenue": round(pred_val, 2),
        "novelty_detected": novelty,
    }
