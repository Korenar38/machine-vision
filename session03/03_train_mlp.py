"""Шаг 3. Многослойная сеть учится узнавать все 10 цифр.

Обучение — это повторение четырёх действий тысячи раз:
посмотреть → посчитать ошибку → найти виноватые веса → чуть-чуть их подправить.
"""

import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from common import MLP, MODEL_FILE, accuracy, mnist, show

torch.manual_seed(0)  # чтобы у всех в классе получались похожие числа

train_loader = DataLoader(mnist(train=True), batch_size=64, shuffle=True)  # пачки по 64 картинки
test_set = mnist(train=False)

model = MLP()
print(model)
print("Весов в сети:", sum(p.numel() for p in model.parameters()))
print(f"Точность до обучения: {accuracy(model, test_set):.1%} — просто угадывает\n")

loss_fn = nn.CrossEntropyLoss()                            # мера ошибки
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # кто подправляет веса

EPOCHS = 3  # эпоха = один проход по всем 60 000 картинкам
history = []
start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    for step, (images, labels) in enumerate(train_loader):
        predictions = model(images)           # 1. посмотреть
        loss = loss_fn(predictions, labels)   # 2. посчитать ошибку
        optimizer.zero_grad()
        loss.backward()                       # 3. найти, какой вес в чём виноват
        optimizer.step()                      # 4. чуть-чуть подправить веса
        if step % 50 == 0:
            history.append(loss.item())
    print(f"Эпоха {epoch}: ошибка {loss.item():.3f}, "
          f"точность на проверке {accuracy(model, test_set):.1%}  ({time.time() - start:.0f} с)")

torch.save(model.state_dict(), MODEL_FILE)
print(f"\nСеть сохранена в {MODEL_FILE.name}")

plt.figure(figsize=(7, 3.8))
plt.plot(history)
plt.xlabel("шаги обучения (×50 пачек)")
plt.ylabel("ошибка")
plt.title("Как падает ошибка во время обучения")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=100)
show()
