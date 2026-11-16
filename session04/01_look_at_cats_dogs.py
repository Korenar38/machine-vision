"""Шаг 1. Знакомимся с данными: кошки и собаки из CIFAR-10.

Цветные картинки 32×32: 10 000 для обучения и 2 000 для проверки.
"""

import matplotlib.pyplot as plt

from common import CLASSES, CLASSES_RU, CatsDogs, show

train = CatsDogs(train=True)   # читается из локальной папки data/cifar-10-batches-py
test = CatsDogs(train=False)

print("Картинок для обучения:", len(train))
print("Картинок для проверки:", len(test))
for label, count in enumerate(train.labels.bincount().tolist()):
    print(f"  {CLASSES_RU[label]}: {count}")

image, label = train[0]
print("Форма картинки:", tuple(image.shape), "— 3 канала, 32×32 пикселя")
print("Всего чисел:", image.numel())
print("Это:", CLASSES_RU[label])
print("Левый верхний пиксель (R, G, B):", [round(v, 2) for v in image[:, 0, 0].tolist()])

fig, axes = plt.subplots(4, 8, figsize=(10, 5.6))
for i, ax in enumerate(axes.flat):
    image, label = train[i]
    ax.imshow(image.permute(1, 2, 0))  # (3, 32, 32) → (32, 32, 3) для matplotlib
    ax.set_title(CLASSES[label])
    ax.axis("off")
fig.suptitle("CIFAR-10: кошки (cat) и собаки (dog)")
plt.tight_layout()
plt.savefig("cats_dogs_grid.png", dpi=100)
show()

image, label = train[1]
fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
axes[0].imshow(image.permute(1, 2, 0), interpolation="nearest")
axes[0].set_title(f"{CLASSES[label]}: RGB")
for ax, channel, name, cmap in zip(axes[1:], image, ["R", "G", "B"], ["Reds", "Greens", "Blues"]):
    ax.imshow(channel, cmap=cmap, interpolation="nearest")
    ax.set_title(f"канал {name}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("one_image.png", dpi=100)
show()
