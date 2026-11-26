from flask import Flask, jsonify, request

from app.ingest import load_raw_transactions
from app.logger import log_event, read_logs
from app.model import load_artifact, predict_revenue, train_model


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        {
            "DATA_DIR": "data/cs-train",
            "MODEL_DIR": "models",
            "LOG_PATH": "logs/api_events.jsonl",
        }
    )
    if config:
        app.config.update(config)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "aavail-revenue-forecast-api"}), 200

    @app.post("/train")
    def train():
        body = request.get_json(silent=True) or {}
        data_dir = body.get("data_dir", app.config["DATA_DIR"])
        country = body.get("country", "all")

        try:
            raw_df = load_raw_transactions(data_dir)
            summary = train_model(raw_df, model_dir=app.config["MODEL_DIR"], country=country)
            log_event(
                log_path=app.config["LOG_PATH"],
                event_type="train",
                payload={"country": country, "data_dir": data_dir, "summary": summary},
                novelty_flag=False,
                metric_value=summary.get("metrics", {}).get("model_rmse"),
            )
            return jsonify({"status": "trained", "result": summary}), 200
        except Exception as exc:
            log_event(
                log_path=app.config["LOG_PATH"],
                event_type="train_error",
                payload={"country": country, "error": str(exc)},
                novelty_flag=True,
            )
            return jsonify({"error": str(exc)}), 400

    @app.post("/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        target_date = body.get("target_date") or body.get("date")
        country = body.get("country", "all")

        if not target_date:
            return jsonify({"error": "El campo 'target_date' es obligatorio."}), 400

        try:
            artifact = load_artifact(app.config["MODEL_DIR"], country=country)
            result = predict_revenue(artifact, target_date=target_date, country=country)
            log_event(
                log_path=app.config["LOG_PATH"],
                event_type="predict",
                payload=result,
                novelty_flag=bool(result.get("novelty_detected", False)),
                metric_value=float(result.get("predicted_revenue", 0.0)),
            )
            return jsonify(result), 200
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/logs")
    def logs():
        limit = request.args.get("limit", default=50, type=int)
        entries = read_logs(app.config["LOG_PATH"], limit=limit)
        novelty_count = sum(1 for item in entries if item.get("novelty_flag"))
        return jsonify(
            {
                "count": len(entries),
                "novelty_events": novelty_count,
                "logs": entries,
            }
        ), 200

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
