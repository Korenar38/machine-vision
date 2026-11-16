"""Шаг 4. Где сеть ошибается? Таблица путаницы и галерея ошибок."""

import matplotlib.pyplot as plt
import torch

from common import load_model, mnist, predict_all, show

model = load_model()
test = mnist(train=False)

predicted = predict_all(model, test)
truth = test.targets
wrong = (predicted != truth).nonzero().flatten()
print(f"Ошибок: {len(wrong)} из {len(test)}\n")

# Таблица путаницы: строка — правильная цифра, столбец — ответ сети
confusion = torch.bincount(truth * 10 + predicted, minlength=100).reshape(10, 10)
print("верно\\сеть " + "".join(f"{d:>5}" for d in range(10)))
for digit in range(10):
    print(f"{digit:>10} " + "".join(f"{n:>5}" for n in confusion[digit].tolist()))

# Самые частые путаницы (не считая диагонали — там верные ответы)
off = confusion.clone()
off.fill_diagonal_(0)
print("\nЧаще всего сеть путает:")
for index in off.flatten().argsort(descending=True)[:3].tolist():
    print(f"  {index // 10} → {index % 10}: {off.flatten()[index].item()} раз")

fig, axes = plt.subplots(3, 8, figsize=(10, 4.4))
for ax, i in zip(axes.flat, wrong[:24].tolist()):
    ax.imshow(test[i][0][0], cmap="gray")
    ax.set_title(f"{truth[i].item()} → {predicted[i].item()}", color="#B8460A")
    ax.axis("off")
fig.suptitle("Ошибки сети: правильная цифра → ответ сети")
plt.tight_layout()
plt.savefig("mistakes.png", dpi=100)
show()
