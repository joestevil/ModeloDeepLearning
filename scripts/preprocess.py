"""
Script de ejemplo para extraer frames desde videos y preparar datos.
Uso básico:
    python scripts/preprocess.py --input "data/raw/video.mp4" --out "data/processed/video_frames" --fps 1
"""
import os
import cv2
import argparse


def extract_frames(video_path, out_dir, fps=1):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se puede abrir video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(round(video_fps / fps)))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extraer frames desde un video')
    parser.add_argument('--input', required=True, help='Ruta al video')
    parser.add_argument('--out', required=True, help='Directorio de salida para frames')
    parser.add_argument('--fps', type=float, default=1.0, help='Frames por segundo a extraer (ej: 1)')
    args = parser.parse_args()
    n = extract_frames(args.input, args.out, fps=args.fps)
    print(f'Frames guardados: {n}')
