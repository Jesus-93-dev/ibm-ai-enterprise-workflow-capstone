$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."
pytest -v tests
