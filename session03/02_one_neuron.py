"""Шаг 2. Один нейрон на чистом NumPy: отличаем 0 от 1.

Нейрон = 784 веса (по одному на пиксель) + смещение.
Он складывает пиксели с весами и «сжимает» сумму в число от 0 до 1:
близко к 1 — «это единица», близко к 0 — «это ноль».
"""

import matplotlib.pyplot as plt
import numpy as np

from common import mnist, show


def zeros_and_ones(dataset):
    """Оставить только нули и единицы; картинки вытянуть в строки по 784 числа."""
    x = dataset.data.numpy().reshape(-1, 784) / 255
    y = dataset.targets.numpy()
    keep = (y == 0) | (y == 1)
    return x[keep], y[keep]


x, y = zeros_and_ones(mnist(train=True))
x_test, y_test = zeros_and_ones(mnist(train=False))
print("Картинок с нулями и единицами:", len(x))

w = np.zeros(784)  # веса: сначала нейрон ничего не знает
b = 0.0            # смещение


def neuron(x):
    return 1 / (1 + np.exp(-(x @ w + b)))  # взвешенная сумма → число от 0 до 1


LEARNING_RATE = 0.5
for step in range(101):
    error = neuron(x) - y                       # насколько ошиблись на каждой картинке
    w -= LEARNING_RATE * x.T @ error / len(x)   # подвинуть каждый вес против ошибки
    b -= LEARNING_RATE * error.mean()
    if step % 20 == 0:
        correct = ((neuron(x_test) > 0.5) == y_test).mean()
        print(f"Шаг {step:3d}: верно на проверке {correct:.1%}")

# Веса можно нарисовать как картинку 28×28 — это «шаблон», который выучил нейрон
limit = np.abs(w).max()
plt.figure(figsize=(5.5, 4.6))
plt.imshow(w.reshape(28, 28), cmap="bwr", vmin=-limit, vmax=limit)
plt.colorbar()
plt.title("Веса нейрона\nкрасное — «за единицу», синее — «за ноль»")
plt.axis("off")
plt.tight_layout()
plt.savefig("neuron_weights.png", dpi=100)
show()
