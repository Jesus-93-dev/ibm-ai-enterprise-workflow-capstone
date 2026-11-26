import glob
import json
from pathlib import Path
import pandas as pd


def load_raw_transactions(data_dir: str) -> pd.DataFrame:
    path = Path(data_dir)
    files = sorted(glob.glob(str(path / "*.json")))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos JSON en: {data_dir}")

    rows = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            if isinstance(payload, list):
                rows.extend(payload)
            elif isinstance(payload, dict):
                rows.append(payload)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if {"year", "month", "day"}.issubset(df.columns):
        date_series = pd.to_datetime(
            {
                "year": pd.to_numeric(df["year"], errors="coerce"),
                "month": pd.to_numeric(df["month"], errors="coerce"),
                "day": pd.to_numeric(df["day"], errors="coerce"),
            },
            errors="coerce",
        )
        df = df.assign(date=date_series.dt.normalize())
    elif "date" in df.columns:
        df = df.assign(date=pd.to_datetime(df["date"], errors="coerce").dt.normalize())
    else:
        raise ValueError("El conjunto de datos no contiene columnas de fecha válidas.")

    price_col = "price" if "price" in df.columns else ("revenue" if "revenue" in df.columns else None)
    revenue_series = pd.to_numeric(df[price_col], errors="coerce").fillna(0.0) if price_col else 0.0

    views_col = "times_viewed" if "times_viewed" in df.columns else ("total_views" if "total_views" in df.columns else None)
    views_series = pd.to_numeric(df[views_col], errors="coerce").fillna(1.0) if views_col else 1.0

    country_series = df["country"].fillna("unknown").astype(str) if "country" in df.columns else "unknown"

    cleaned = df.assign(
        revenue=revenue_series,
        total_views=views_series,
        purchases=1,
        country=country_series,
    ).dropna(subset=["date"])

    return cleaned.reset_index(drop=True)


def aggregate_daily(df: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "revenue", "purchases", "total_views"])

    working = df.copy()
    if "date" not in working.columns and {"year", "month", "day"}.issubset(working.columns):
        working["date"] = pd.to_datetime(
            {
                "year": pd.to_numeric(working["year"], errors="coerce"),
                "month": pd.to_numeric(working["month"], errors="coerce"),
                "day": pd.to_numeric(working["day"], errors="coerce"),
            },
            errors="coerce",
        ).dt.normalize()

    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    working = working.dropna(subset=["date"])

    if "revenue" not in working.columns:
        price_col = "price" if "price" in working.columns else None
        working["revenue"] = pd.to_numeric(working[price_col], errors="coerce").fillna(0.0) if price_col else 0.0
    else:
        working["revenue"] = pd.to_numeric(working["revenue"], errors="coerce").fillna(0.0)

    if "total_views" not in working.columns:
        views_col = "times_viewed" if "times_viewed" in working.columns else None
        working["total_views"] = pd.to_numeric(working[views_col], errors="coerce").fillna(1.0) if views_col else 1.0
    else:
        working["total_views"] = pd.to_numeric(working["total_views"], errors="coerce").fillna(0.0)

    if "purchases" not in working.columns:
        working["purchases"] = 1
    else:
        working["purchases"] = pd.to_numeric(working["purchases"], errors="coerce").fillna(1).astype(int)

    if country and country.lower() not in {"all", "global", "combined"}:
        if "country" in working.columns:
            working = working[working["country"].astype(str).str.lower() == country.lower()]

    if working.empty:
        return pd.DataFrame(columns=["date", "revenue", "purchases", "total_views"])

    grouped = (
        working.groupby("date", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            purchases=("purchases", "sum"),
            total_views=("total_views", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    full_idx = pd.date_range(grouped["date"].min(), grouped["date"].max(), freq="D")
    grouped = (
        grouped.set_index("date")
        .reindex(full_idx)
        .rename_axis("date")
        .reset_index()
    )
    grouped["revenue"] = grouped["revenue"].fillna(0.0)
    grouped["purchases"] = grouped["purchases"].fillna(0).astype(int)
    grouped["total_views"] = grouped["total_views"].fillna(0.0)
    return grouped


def build_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.sort_values("date").reset_index(drop=True).copy()
    df["lag_1"] = df["revenue"].shift(1)
    df["lag_7"] = df["revenue"].shift(7)
    df["lag_14"] = df["revenue"].shift(14)
    df["roll_mean_7"] = df["revenue"].shift(1).rolling(7).mean()
    df["roll_mean_30"] = df["revenue"].shift(1).rolling(30).mean()
    df["views_lag_1"] = df["total_views"].shift(1)
    df["purchases_lag_1"] = df["purchases"].shift(1)
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["target"] = df["revenue"].shift(-30)
    return df
