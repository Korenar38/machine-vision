"""Шаг 1. Знакомимся с MNIST: 70 000 рукописных цифр 28×28."""

import matplotlib.pyplot as plt

from common import mnist, show

train = mnist(train=True)   # читается из локальной папки data/MNIST
test = mnist(train=False)

print("Картинок для обучения:", len(train))
print("Картинок для проверки:", len(test))

img, label = train[0]
print("Форма одной картинки:", tuple(img.shape), "— 1 канал, 28×28 пикселей")
print("Это цифра:", label)
print()

# Картинка — таблица чисел от 0 до 1. Нарисуем её символами прямо в консоли
for row in img[0]:
    print("".join("██" if v > 0.5 else "··" for v in row))

fig, axes = plt.subplots(4, 8, figsize=(10, 5.6))
for i, ax in enumerate(axes.flat):
    image, digit = train[i]
    ax.imshow(image[0], cmap="gray")
    ax.set_title(digit)
    ax.axis("off")
fig.suptitle("MNIST: первые 32 картинки и правильные ответы")
plt.tight_layout()
plt.savefig("mnist_grid.png", dpi=100)
show()
