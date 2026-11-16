"""Шаг 7. Кошка или собака? Проверяем сеть на своих фотографиях.

Положите свои фото (jpg/png) в папку my_photos рядом с этим файлом.
Если папки нет, сеть проверяется на случайных картинках из проверочного набора.
Нужна обученная сеть cnn_aug.pt (шаг 5).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as TF

from common import CLASSES, CLASSES_RU, CatsDogs, SimpleCNN, load_model, show

model = load_model(SimpleCNN(dropout=0.3), "cnn_aug")

PHOTOS = Path("my_photos")
files = sorted(p for p in PHOTOS.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
print("Своих фото:", len(files))

def prepare(photo):
    """Фото любого размера → тензор (3, 32, 32), числа 0..1, как картинки CIFAR."""
    x = TF.pil_to_tensor(photo)                      # (3, H, W), байты 0..255
    x = TF.center_crop(x, min(x.shape[1:]))          # квадрат из центра
    x = TF.resize(x, [32, 32], antialias=True)       # 32×32
    return x.float() / 255


if files:
    originals = [Image.open(p).convert("RGB") for p in files]
    names = [p.name for p in files]
    small = torch.stack([prepare(photo) for photo in originals])
else:
    print("Папки my_photos нет или она пустая — берём случайные картинки из проверочного набора")
    test = CatsDogs(train=False)
    picks = torch.randint(len(test), (8,)).tolist()
    small = test.images[picks]
    originals = [img.permute(1, 2, 0) for img in small]
    names = [f"test #{i}: {CLASSES[test.labels[i]]}" for i in picks]
print("Пачка для сети:", tuple(small.shape))

with torch.no_grad():
    probs = torch.softmax(model(small), dim=1)

for name, p in zip(names, probs):
    best = p.argmax().item()
    print(f"{name:<28} → {CLASSES_RU[best]:<7} ({p[best].item():.0%})")

n = len(names)
fig, axes = plt.subplots(2, n, figsize=(max(1.6 * n + 1, 6), 3.8), squeeze=False)
for col, (original, tiny, p, name) in enumerate(zip(originals, small, probs, names)):
    best = p.argmax().item()
    axes[0, col].imshow(original)
    axes[0, col].set_title(name, fontsize=8)
    axes[1, col].imshow(tiny.permute(1, 2, 0), interpolation="nearest")
    axes[1, col].set_title(f"{CLASSES[best]} {p[best].item():.0%}", color="#B8460A")
for ax in axes.flat:
    ax.axis("off")
fig.suptitle("Сверху — фото, снизу — что видит сеть и её ответ")
plt.tight_layout()
plt.savefig("my_photos_result.png", dpi=100)
show()
