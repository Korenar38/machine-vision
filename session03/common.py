"""Общие части сессии 3: данные MNIST, модель и проверка точности."""

import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA = Path(__file__).parent.parent / "data"
MODEL_FILE = Path(__file__).parent / "mlp.pt"


def mnist(train=True):
    """Вернуть MNIST из локальной папки data/MNIST: картинки 1×28×28, числа от 0 до 1."""
    if not (DATA / "MNIST" / "raw").exists():
        raise SystemExit(f"Нет набора MNIST: положите папку MNIST/raw в {DATA}")
    return datasets.MNIST(DATA, train=train, download=False, transform=transforms.ToTensor())


class MLP(nn.Module):
    """Многослойная сеть: 784 входа → 128 → 64 → 10 выходов (по одному на цифру)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),                    # 28×28 → 784 числа в одну строку
            nn.Linear(784, 128), nn.ReLU(),  # слой 1: 128 нейронов
            nn.Linear(128, 64), nn.ReLU(),   # слой 2: 64 нейрона
            nn.Linear(64, 10),               # выход: 10 оценок, по одной на цифру
        )

    def forward(self, x):
        return self.layers(x)


def train(model, dataset, epochs=3):
    """Тот же цикл обучения, что в 03_train_mlp.py, только без пояснений."""
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_all(model, dataset):
    """Ответы сети на весь набор: номер цифры для каждой картинки."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1000)
    return torch.cat([model(images).argmax(1) for images, _ in loader])


def accuracy(model, dataset):
    """Доля картинок, на которых сеть ответила верно."""
    if hasattr(dataset, "targets"):
        truth = dataset.targets
    else:
        truth = torch.tensor([label for _, label in dataset])
    return (predict_all(model, dataset) == truth).float().mean().item()


def load_model():
    """Загрузить обученную сеть из файла mlp.pt."""
    if not MODEL_FILE.exists():
        raise SystemExit("Сначала обучите сеть: python 03_train_mlp.py")
    model = MLP()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    return model


def show():
    """Показать графики matplotlib (в тихом режиме NO_GUI только сохраняем)."""
    import matplotlib.pyplot as plt
    if not os.environ.get("NO_GUI"):
        plt.show()
    plt.close("all")
