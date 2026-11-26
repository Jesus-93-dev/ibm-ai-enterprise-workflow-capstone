$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."
python -m pytest -v tests
