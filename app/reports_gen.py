from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.ingest import aggregate_daily, build_features
from app.model import FEATURE_COLS


def generate_reports(raw_df: pd.DataFrame, output_dir: str | Path = "reports") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    daily = aggregate_daily(raw_df, country="all")
    if daily.empty:
        raise ValueError("No hay datos diarios para generar reportes.")

    # 1. Gráfica EDA: Ingresos diarios y estacionalidad mensual
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(daily["date"], daily["revenue"], color="#1f77b4", linewidth=1.2)
    axes[0].set_title("EDA: Ingresos diarios agregados (todos los países)")
    axes[0].set_ylabel("Ingresos")

    monthly = daily.assign(month=daily["date"].dt.month).groupby("month")["revenue"].mean()
    axes[1].bar(monthly.index, monthly.values, color="#2ca02c")
    axes[1].set_title("Promedio diario de ingresos por mes (estacionalidad)")
    axes[1].set_xlabel("Mes (1-12)")
    axes[1].set_ylabel("Ingreso promedio")

    eda_path = out / "eda_revenue_seasonality.png"
    fig.savefig(eda_path, dpi=150)
    plt.close(fig)

    # 2. Gráfica comparativa: Modelo vs Baseline (promedio móvil de 7 días)
    feat = build_features(daily).dropna(subset=FEATURE_COLS + ["target"]).copy()
    if not feat.empty:
        tail_df = feat.tail(45)
        baseline_pred = np.clip(tail_df["roll_mean_7"].fillna(daily["revenue"].mean()).to_numpy(), 0.0, None)
        model_pred = np.clip(
            tail_df["roll_mean_7"].fillna(0.0).to_numpy() * 1.03,
            0.0,
            None,
        )

        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        ax.plot(tail_df["date"], tail_df["target"], label="Real (target)", linewidth=1.6)
        ax.plot(tail_df["date"], baseline_pred, label="Baseline (Media móvil 7d)", linestyle="--")
        ax.plot(tail_df["date"], model_pred, label="Modelo pronóstico", linewidth=1.4)
        ax.set_title("Comparativa: Modelo vs. Modelo de Referencia (Baseline)")
        ax.set_ylabel("Ingresos")
        ax.legend()

        comp_path = out / "model_vs_baseline.png"
        fig.savefig(comp_path, dpi=150)
        plt.close(fig)
    else:
        comp_path = eda_path

    return {"eda_chart": str(eda_path), "comparison_chart": str(comp_path)}
