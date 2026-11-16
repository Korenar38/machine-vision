"""Шаг 4. Простая свёрточная сеть учится отличать кошек от собак.

Три блока «свёртка → ReLU → пулинг» + классификатор. Сохраняет cnn.pt и cnn.json.
"""

import matplotlib.pyplot as plt
import torch

from common import MLP, CatsDogs, SimpleCNN, count_params, load_history, save_model, show, train

torch.manual_seed(0)
train_set, test_set = CatsDogs(train=True), CatsDogs(train=False)

model = SimpleCNN()
print(model)
print("Весов в свёрточной сети:", count_params(model))
print("Весов в MLP из шага 3:  ", count_params(MLP()))

x = torch.zeros(1, 3, 32, 32)
print(f"{'вход':>10}: {tuple(x.shape)}")
for layer in model.features:
    x = layer(x)
    print(f"{type(layer).__name__:>10}: {tuple(x.shape)}")

history = train(model, train_set, test_set, epochs=25)
save_model(model, "cnn", history)

epochs = range(1, len(history["train"]) + 1)
plt.figure(figsize=(7, 3.8))
plt.plot(epochs, history["train"], "o-", label="CNN: обучающие")
plt.plot(epochs, history["test"], "o-", label="CNN: проверочные")
mlp = load_history("mlp")
if mlp:
    plt.plot(range(1, len(mlp["test"]) + 1), mlp["test"], "s--", label="MLP: проверочные")
plt.axhline(0.5, color="gray", linestyle=":", label="угадывание (50%)")
plt.xlabel("эпоха")
plt.ylabel("точность")
plt.title("Свёрточная сеть против полносвязной")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cnn_curves.png", dpi=100)
show()
