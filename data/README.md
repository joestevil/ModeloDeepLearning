Carpeta data/

- `raw/` : colocar aquí los videos sin procesar (NO versionar en git si son grandes).
- `processed/` : outputs del preprocesado (frames, clips, crops).
- `annotations/` : archivos de anotaciones (COCO JSON, YOLO txt, CSV) con las etiquetas y timestamps.

Formato sugerido de anotaciones:

- Para clasificación por clip: CSV con columnas [video, start_time, end_time, label]
- Para detección/segmentación por frame: COCO JSON o YOLO per-frame

Ejemplo de flujo:

1. `scripts/preprocess.py --input data/raw/video.mp4 --out data/processed/video_frames/`
2. Crear/convertir anotaciones en `data/annotations/`.
