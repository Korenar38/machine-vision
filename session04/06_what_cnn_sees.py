"""Шаг 6. Заглядываем внутрь: фильтры, карты признаков и ошибки свёрточной сети.

Нужна обученная сеть cnn_aug.pt (шаг 5).
"""

import matplotlib.pyplot as plt
import torch

from common import CLASSES, CLASSES_RU, CatsDogs, SimpleCNN, accuracy, load_model, predict_all, show

model = load_model(SimpleCNN(dropout=0.3), "cnn_aug")
test = CatsDogs(train=False)
print(f"Точность на проверке: {accuracy(model, test):.1%}")

weights = model.features[0].weight.detach()          # (16, 3, 3, 3)
fig, axes = plt.subplots(2, 8, figsize=(9, 2.6))
for ax, w in zip(axes.flat, weights):
    w = (w - w.min()) / (w.max() - w.min())          # веса → 0..1, чтобы показать как цвет
    ax.imshow(w.permute(1, 2, 0), interpolation="nearest")
    ax.axis("off")
fig.suptitle("16 фильтров первого слоя (3×3, цветные)")
plt.tight_layout()
plt.savefig("filters.png", dpi=100)
show()

image, label = test[0]
maps = []
x = image[None]
with torch.no_grad():
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, torch.nn.MaxPool2d):   # конец блока «свёртка → ReLU → пулинг»
            maps.append(x[0])

fig, axes = plt.subplots(3, 9, figsize=(12, 4.4))
for row, block in zip(axes, maps):
    row[0].imshow(image.permute(1, 2, 0))
    row[0].set_title(CLASSES[label])
    for ax, feature_map in zip(row[1:], block):
        ax.imshow(feature_map, cmap="viridis", interpolation="nearest")
    row[1].set_title(f"{block.shape[1]}x{block.shape[2]}, {block.shape[0]} maps", loc="left")
for ax in axes.flat:
    ax.axis("off")
fig.suptitle("Карты признаков после блоков 1, 2 и 3")
plt.tight_layout()
plt.savefig("feature_maps.png", dpi=100)
show()

probs = predict_all(model, test)
predicted = probs.argmax(1)
truth = test.labels

confusion = torch.bincount(truth * 2 + predicted, minlength=4).reshape(2, 2)
print("верно\\сеть    cat    dog")
for i, name in enumerate(CLASSES):
    print(f"{name:>10} {confusion[i, 0].item():>6} {confusion[i, 1].item():>6}")
for i, name in enumerate(CLASSES_RU):
    print(f"Точность на классе «{name}»: {confusion[i, i].item() / confusion[i].sum().item():.1%}")

wrong = (predicted != truth).nonzero().flatten()
confidence = probs[wrong].max(1).values
worst = wrong[confidence.argsort(descending=True)[:16]]

fig, axes = plt.subplots(2, 8, figsize=(11, 3.4))
for ax, i in zip(axes.flat, worst.tolist()):
    ax.imshow(test[i][0].permute(1, 2, 0))
    ax.set_title(f"{CLASSES[truth[i]]}→{CLASSES[predicted[i]]}\n{probs[i].max().item():.0%}",
                 color="#B8460A", fontsize=9)
    ax.axis("off")
fig.suptitle("Самые уверенные ошибки: правильный ответ → ответ сети, уверенность")
plt.tight_layout()
plt.savefig("cnn_mistakes.png", dpi=100)
show()
