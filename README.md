# Proyecto: Detección y Clasificación de Robos/Hurtos/Peleas en Videovigilancia

Descripción:
Este repositorio contiene la estructura inicial para entrenar modelos de Deep Learning destinados a detección y clasificación de incidentes (robos, hurtos, peleas) en video de vigilancia.

Estructura principal:

- `data/` - datos de entrada
  - `raw/` - videos originales (no versionados)
  - `processed/` - frames, clips y datos preparados
  - `annotations/` - archivos de anotaciones (COCO, YOLO, CSV, etc.)
- `src/` - código fuente
  - `data/` - loaders, transformaciones
  - `models/` - definiciones de redes
  - `training/` - scripts de entrenamiento y evaluación
  - `utils/` - utilidades (logging, metrics)
- `notebooks/` - notebooks para exploración y pruebas
- `experiments/` - registros y configuraciones de experimentos
- `models/` - checkpoints guardados (no commitear pesos grandes)
- `scripts/` - scripts de utilidad (preprocesado, conversión)
- `tests/` - pruebas unitarias mínimas
- `docs/` - documentación adicional

Primeros pasos:

1. Coloca los videos en `data/raw/`.
2. Añade anotaciones en `data/annotations/` (formato COCO o YOLO recomendado).
3. Ejecuta `python scripts/preprocess.py --help` para ver opciones de preprocesado.
4. Ajusta `requirements.txt`, crea un entorno virtual e instala dependencias.

Nota sobre datos sensibles:
Asegúrate de cumplir la normativa local y la privacidad al recolectar/usar videos de vigilancia.

Licencia: Añade aquí la licencia que prefieras.
