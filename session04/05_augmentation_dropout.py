"""Шаг 5. Боремся с переобучением: аугментация данных и dropout.

Та же SimpleCNN, но картинки при обучении случайно отражаются, сдвигаются и меняют яркость,
а часть нейронов классификатора случайно выключается. Сохраняет cnn_aug.pt и cnn_aug.json.
"""

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as T

from common import CLASSES, CatsDogs, SimpleCNN, load_history, save_model, show, train

augment = T.Compose([
    T.RandomHorizontalFlip(),                                      # зеркало
    T.RandomCrop(32, padding=4, padding_mode="reflect"),           # сдвиг до 4 пикселей
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),   # освещение
])

torch.manual_seed(0)
plain = CatsDogs(train=True)
image, label = plain[3]

fig, axes = plt.subplots(1, 9, figsize=(12, 1.9))
axes[0].imshow(image.permute(1, 2, 0))
axes[0].set_title(f"original: {CLASSES[label]}")
for ax in axes[1:]:
    ax.imshow(augment(image).clamp(0, 1).permute(1, 2, 0))
    ax.set_title("augmented")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig("augment.png", dpi=100)
show()

torch.manual_seed(0)
train_set = CatsDogs(train=True, transform=augment)
test_set = CatsDogs(train=False)

model = SimpleCNN(dropout=0.3)
history = train(model, train_set, test_set, epochs=40)
save_model(model, "cnn_aug", history)

plt.figure(figsize=(7, 3.8))
plain_cnn = load_history("cnn")
if plain_cnn:
    e = range(1, len(plain_cnn["train"]) + 1)
    plt.plot(e, plain_cnn["train"], "o-", color="C0", label="шаг 4: обучающие")
    plt.plot(e, plain_cnn["test"], "o-", color="C1", label="шаг 4: проверочные")
e = range(1, len(history["train"]) + 1)
plt.plot(e, history["train"], "s--", color="C0", label="аугментация: обучающие")
plt.plot(e, history["test"], "s--", color="C1", label="аугментация: проверочные")
plt.xlabel("эпоха")
plt.ylabel("точность")
plt.title("Аугментация и dropout против переобучения")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("aug_curves.png", dpi=100)
show()
