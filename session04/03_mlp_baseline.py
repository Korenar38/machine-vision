"""Шаг 3. Точка отсчёта: полносвязная сеть из сессии 3 на кошках и собаках.

Сохраняет историю обучения в mlp.json — с ней сравним свёрточную сеть.
"""

import matplotlib.pyplot as plt
import torch

from common import MLP, CatsDogs, count_params, save_model, show, train

torch.manual_seed(0)
train_set, test_set = CatsDogs(train=True), CatsDogs(train=False)

model = MLP()
print(model)
print("Весов в сети:", count_params(model))

history = train(model, train_set, test_set, epochs=8)
save_model(model, "mlp", history)

epochs = range(1, len(history["train"]) + 1)
plt.figure(figsize=(7, 3.8))
plt.plot(epochs, history["train"], "o-", label="обучающие")
plt.plot(epochs, history["test"], "o-", label="проверочные")
plt.axhline(0.5, color="gray", linestyle="--", label="угадывание (50%)")
plt.xlabel("эпоха")
plt.ylabel("точность")
plt.title("Полносвязная сеть: кошки и собаки")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("mlp_curves.png", dpi=100)
show()
