# Pronóstico de ingresos para AAVAIL - IBM AI Enterprise Workflow Capstone

Solución modular en Python para ingesta de series temporales diarias, entrenamiento y predicción de ingresos (por país y global), API en Flask con registro (logging) y detección de novedades, pruebas unitarias aisladas y despliegue con Docker.

## Estructura del proyecto
- app/ingest.py: carga de archivos JSON, agregación diaria por país o global y generación de características/objetivo.
- app/model.py: entrenamiento, evaluación frente a baseline, guardado/carga de artefactos y predicción con detección de novedades.
- app/logger.py: registro estructurado en JSONL para auditoría, métricas y novelty flags.
- app/api.py: endpoints Flask (/health, /train, /predict, /logs).
- app/reports_gen.py: generación de gráficas de EDA y comparativa frente a baseline.
- tests/: pruebas unitarias aisladas usando directorios temporales (tmp_path).
- run_tests.ps1: ejecución unificada de todas las pruebas unitarias.
- Dockerfile: empaquetado de la aplicación y pruebas.

## Ejecución de pruebas unitarias
.\run_tests.ps1

## Levantar API con Docker
docker build -t aavail-capstone-api .
docker run -p 5000:5000 aavail-capstone-api
