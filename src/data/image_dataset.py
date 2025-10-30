"""Dataset sencillo para imágenes de clasificación en carpetas por clase.

Estructura esperada:
root/
  class_0/
    img1.jpg
    img2.jpg
  class_1/
    img3.jpg

Devuelve (image_tensor, label) con shape [C,H,W].
"""
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms


class ImageFolderDataset(Dataset):
    """Dataset simple tipo ImageFolder. Si tus datos ya vienen por frames,
    organiza cada clase en una carpeta bajo `root`.

    Args:
        root (str): directorio raíz con subcarpetas por clase.
        transform (callable, optional): transformaciones a aplicar a PIL.Image.
    """
    def __init__(self, root, transform=None):
        self.samples = []
        root = os.path.abspath(root)
        for label_name in sorted(os.listdir(root)):
            label_dir = os.path.join(root, label_name)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(label_dir, fname), label_name))

        if len(self.samples) == 0:
            raise RuntimeError(f'No se encontraron imágenes en: {root}')

        self.classes = sorted(list({s[1] for s in self.samples}))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_name = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = self.class_to_idx[label_name]
        return img, label


if __name__ == '__main__':
    # pequeño test manual
    from torchvision import transforms
    ds = ImageFolderDataset('data/processed/images_example', transform=transforms.Compose([
        transforms.Resize((64, 64)), transforms.ToTensor()
    ]))
    print('Samples:', len(ds))
    x, y = ds[0]
    print('Shape:', x.shape, 'Label:', y)
