"""Шаг 2. Свёртка своими руками: фильтр 3×3 скользит по картинке и ищет узор.

Ручные фильтры (границы, размытие, резкость), ReLU, пулинг и слой nn.Conv2d.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from common import CatsDogs, show

test = CatsDogs(train=False)
cat_index = (test.labels == 0).nonzero()[0].item()  # первая кошка в проверочном наборе
image, _ = test[cat_index]
gray = image.mean(0, keepdim=True)  # (3, 32, 32) → (1, 32, 32): среднее трёх каналов
print("Серая картинка:", tuple(gray.shape))

sobel_x = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]])
KERNELS = {
    "vertical edges": sobel_x,
    "horizontal edges": sobel_x.T,
    "blur": torch.ones(3, 3) / 9,
    "sharpen": torch.tensor([[0., -1., 0.],
                             [-1., 5., -1.],
                             [0., -1., 0.]]),
}

y, x = 16, 16
patch = gray[0, y - 1:y + 2, x - 1:x + 2]           # окно 3×3 вокруг точки
by_hand = (patch * sobel_x).sum().item()            # 9 умножений и сумма
result = F.conv2d(gray[None], sobel_x[None, None], padding=1)
print("Окно 3×3:\n", patch.numpy().round(2))
print(f"Вручную: {by_hand:.4f}   F.conv2d: {result[0, 0, y, x].item():.4f}")
print("Размер результата:", tuple(result.shape))

fig, axes = plt.subplots(2, 4, figsize=(11, 5.8))
for ax, (name, kernel) in zip(axes[0], KERNELS.items()):
    out = F.conv2d(gray[None], kernel[None, None], padding=1)[0, 0]
    limit = out.abs().max()
    ax.imshow(out, cmap="gray" if name in ("blur", "sharpen") else "bwr",
              vmin=None if name in ("blur", "sharpen") else -limit,
              vmax=None if name in ("blur", "sharpen") else limit)
    ax.set_title(name)

edges = F.conv2d(gray[None], sobel_x[None, None], padding=1)  # (1, 1, 32, 32)
relu = F.relu(edges)                                          # отрицательное → 0
pooled = F.max_pool2d(relu, 2)                                # 32×32 → 16×16
for ax, picture, title in zip(axes[1],
                              [image.permute(1, 2, 0), gray[0], relu[0, 0], pooled[0, 0]],
                              ["original", "gray", "vertical edges + ReLU", f"max pool 2x2: {tuple(pooled.shape[2:])}"]):
    ax.imshow(picture, cmap="gray", interpolation="nearest")
    ax.set_title(title)
for ax in axes.flat:
    ax.axis("off")
fig.suptitle("Свёртка ручными ядрами (сверху) и блок «свёртка → ReLU → пулинг» (снизу)")
plt.tight_layout()
plt.savefig("conv_by_hand.png", dpi=100)
show()

conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
print("Форма весов:", tuple(conv.weight.shape), "— 16 ядер по 3×3×3")
out = conv(image[None])
print("Вход:", tuple(image[None].shape), "→ выход:", tuple(out.shape))

conv_params = sum(p.numel() for p in conv.parameters())
dense_params = image.numel() * out.numel()
print(f"Весов у свёртки:             {conv_params}")
print(f"Весов у полносвязного слоя:  {dense_params:,}".replace(",", " "))
