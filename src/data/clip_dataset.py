"""Dataset para pequeños clips: soporta clips como carpetas de frames o archivos de vídeo cortos.

Salida: tensor [C, T, H, W], label (placeholder 0 si no hay etiquetas explícitas).
"""
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import transforms

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def read_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'No se puede abrir video: {video_path}')
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        frames.append(img)
    cap.release()
    return frames


class ClipFolderDataset(Dataset):
    """Dataset que acepta:
    - clips_root con subcarpetas (cada subcarpeta -> secuencia de frames .jpg/.png)
    - clips_root con archivos de vídeo (cada archivo -> clip corto)

    Args:
        clips_root (str): ruta que contiene subdirs por clip o vídeo por clip.
        transform (callable): transform to apply per frame (PIL->Tensor).
        seq_len (int): número de frames por muestra (se muestrea o se rellena).
    """
    def __init__(self, clips_root, transform=None, seq_len=16):
        self.clips_root = os.path.abspath(clips_root)
        entries = []
        for name in sorted(os.listdir(self.clips_root)):
            p = os.path.join(self.clips_root, name)
            if os.path.isdir(p):
                entries.append((p, 'dir'))
            elif os.path.isfile(p) and name.lower().endswith(VIDEO_EXTS):
                entries.append((p, 'video'))
        if len(entries) == 0:
            raise RuntimeError(f'No se encontraron clips en: {clips_root}')
        self.entries = entries
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.seq_len = seq_len

    def __len__(self):
        return len(self.entries)

    def _load_from_dir(self, dir_path):
        frames = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        imgs = [Image.open(p).convert('RGB') for p in frames]
        return imgs

    def __getitem__(self, idx):
        path, kind = self.entries[idx]
        if kind == 'dir':
            imgs = self._load_from_dir(path)
        else:
            imgs = read_frames_from_video(path)

        n = len(imgs)
        if n == 0:
            raise RuntimeError(f'Clip vacío en: {path}')

        if n >= self.seq_len:
            indices = np.linspace(0, n - 1, self.seq_len, dtype=int)
            sampled = [imgs[i] for i in indices]
        else:
            sampled = imgs + [imgs[-1]] * (self.seq_len - n)

        if self.transform:
            sampled = [self.transform(im) for im in sampled]

        # stack -> [T, C, H, W] -> [C, T, H, W]
        tensor = torch.stack(sampled).permute(1, 0, 2, 3)
        label = 0  # placeholder: integrar con anotaciones si existen
        return tensor, label


if __name__ == '__main__':
    # pequeño ejemplo de uso
    ds = ClipFolderDataset('data/processed/clips_example', seq_len=8)
    print('Clips:', len(ds))
    x, y = ds[0]
    print('Shape clip:', x.shape)
