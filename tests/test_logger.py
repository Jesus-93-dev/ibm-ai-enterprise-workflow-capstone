from app.logger import log_event, read_logs


def test_logger_isolated(tmp_path):
    log_file = tmp_path / "logs" / "events.jsonl"
    log_event(log_file, "predict", {"country": "all", "predicted_revenue": 250.5}, novelty_flag=True, metric_value=250.5)
    log_event(log_file, "predict", {"country": "France", "predicted_revenue": 80.0}, novelty_flag=False, metric_value=80.0)

    items = read_logs(log_file, limit=10)
    assert len(items) == 2
    assert items[0]["novelty_flag"] is True
    assert items[1]["payload"]["country"] == "France"
