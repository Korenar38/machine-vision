"""Шаг 6. Эксперимент: перемешаем пиксели. Заметит ли сеть?

Все картинки перемешиваются одинаково — как будто кто-то переставил
провода от пикселей к входам сети. Человек не узнает ни одной цифры.
А сеть?
"""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from common import MLP, accuracy, load_model, mnist, show, train

torch.manual_seed(0)
order = torch.randperm(784)  # один и тот же новый порядок пикселей для всех картинок


class Shuffled(Dataset):
    """MNIST, в котором пиксели каждой картинки переставлены по order."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, label = self.base[i]
        return image.view(-1)[order].view(1, 28, 28), label


train_set, test_set = Shuffled(mnist(train=True)), Shuffled(mnist(train=False))

fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
for i in range(6):
    axes[0, i].imshow(test_set.base[i][0][0], cmap="gray")
    axes[1, i].imshow(test_set[i][0][0], cmap="gray")
    axes[0, i].set_title(test_set.base[i][1])
    axes[0, i].axis("off")
    axes[1, i].axis("off")
fig.suptitle("Сверху — обычные цифры, снизу — те же с перемешанными пикселями")
plt.tight_layout()
plt.savefig("shuffled.png", dpi=100)
show()

print("Обучаем такую же сеть на перемешанных картинках (около минуты)...")
model = MLP()
train(model, train_set, epochs=3)

normal = accuracy(load_model(), mnist(train=False))
shuffled = accuracy(model, test_set)
print(f"\nОбычные картинки:      {normal:.1%}")
print(f"Перемешанные пиксели:  {shuffled:.1%}")
print("\nВопрос: что это говорит о том, как сеть «видит» цифру?")
print("И что будет на фото кошек, где форма важнее отдельных пикселей?  → сессия 4")
