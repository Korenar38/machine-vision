"""Общие части сессии 4: кошки и собаки из CIFAR-10, модели и обучение."""

import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

DATA = Path(__file__).parent.parent / "data"
HERE = Path(__file__).parent

CLASSES = ["cat", "dog"]           # 0 — кошка, 1 — собака (латиницей: так подписывает matplotlib без сюрпризов)
CLASSES_RU = ["кошка", "собака"]
CIFAR_CAT, CIFAR_DOG = 3, 5        # номера этих классов в CIFAR-10


class CatsDogs(Dataset):
    """Только кошки и собаки из CIFAR-10: картинки 3×32×32, числа от 0 до 1.

    transform — что сделать с картинкой перед выдачей (например, случайно отразить).
    """

    def __init__(self, train=True, transform=None):
        if not (DATA / "cifar-10-batches-py").exists():
            raise SystemExit(f"Нет набора CIFAR-10: положите папку cifar-10-batches-py в {DATA}")
        cifar = datasets.CIFAR10(DATA, train=train, download=False)  # берём из локальной папки data
        targets = torch.tensor(cifar.targets)
        keep = (targets == CIFAR_CAT) | (targets == CIFAR_DOG)
        # (N, 32, 32, 3) байты → (N, 3, 32, 32) дроби от 0 до 1: так любит PyTorch
        self.images = torch.tensor(cifar.data[keep.numpy()]).permute(0, 3, 1, 2).float() / 255
        self.labels = (targets[keep] == CIFAR_DOG).long()  # кошка → 0, собака → 1
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[i]


class MLP(nn.Module):
    """Сеть из сессии 3, только вход больше: 3×32×32 = 3072 числа."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleCNN(nn.Module):
    """Простая свёрточная сеть: три блока «свёртка → ReLU → пулинг» и классификатор.

    32×32 → 16×16 → 8×8 → 4×4; каналов: 3 → 16 → 32 → 64.
    dropout > 0 включает «выключение» случайных нейронов при обучении (шаг 5).
    """

    def __init__(self, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16×16×16
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32×8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64×4×4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 64×4×4 → 1024 числа
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),                  # две оценки: «кошка» и «собака»
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    """Сколько весов в сети."""
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_all(model, dataset):
    """Вероятности классов для всех картинок набора: тензор (N, 2)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=500)
    return torch.cat([torch.softmax(model(images), dim=1) for images, _ in loader])


def accuracy(model, dataset):
    """Доля картинок, на которых сеть ответила верно."""
    return (predict_all(model, dataset).argmax(1) == dataset.labels).float().mean().item()


def train(model, train_set, test_set, epochs=10, lr=0.001):
    """Цикл обучения из сессии 3 + запись истории точности после каждой эпохи.

    Возвращает словарь {"train": [...], "test": [...]} — точность по эпохам.
    """
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = {"train": [], "test": []}
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        for images, labels in loader:
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (predictions.argmax(1) == labels).sum().item()
        history["train"].append(correct / len(train_set))
        history["test"].append(accuracy(model, test_set))
        print(f"Эпоха {epoch:2d}: на обучающих {history['train'][-1]:.1%}, "
              f"на проверочных {history['test'][-1]:.1%}")
    return history


def save_model(model, name, history=None):
    """Сохранить веса в name.pt и (если есть) историю обучения в name.json."""
    torch.save(model.state_dict(), HERE / f"{name}.pt")
    if history:
        (HERE / f"{name}.json").write_text(json.dumps(history), encoding="utf-8")
    print(f"Сеть сохранена в {name}.pt")


def load_model(model, name):
    """Загрузить веса из name.pt в готовую сеть model."""
    path = HERE / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"Нет файла {path.name} — сначала выполните шаг, который его обучает")
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_history(name):
    """Прочитать историю обучения из name.json (или None, если файла нет)."""
    path = HERE / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def show():
    """Показать графики matplotlib (в тихом режиме NO_GUI только сохраняем)."""
    import matplotlib.pyplot as plt
    if not os.environ.get("NO_GUI"):
        plt.show()
    plt.close("all")
