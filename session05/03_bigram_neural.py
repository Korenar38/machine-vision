"""Шаг 3. Та же биграммная модель, но как нейросеть.

Вместо подсчётов — таблица весов (nn.Embedding), которую подбирает градиентный спуск.
Сеть сама «доходит» до тех же вероятностей, что мы посчитали руками в шаге 2.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from common import BOUNDARY, bigram_counts, bigram_pairs, load_names, name_tokenizer, sample_name, show

torch.manual_seed(0)
names = load_names()
tok = name_tokenizer(names)
xs, ys = bigram_pairs(names, tok)
V = tok.vocab_size

# Embedding(V, V): для каждого символа — строка из V чисел, «сырые оценки» (logits) следующего символа.
# Это то же самое, что Linear(V, V) от вектора one-hot, только быстрее.
model = nn.Embedding(V, V)
print("Весов в сети:", sum(p.numel() for p in model.parameters()), f"= {V} × {V}")

x = F.one_hot(xs[:3], V).float()
print("Первые три буквы как one-hot (одна единица, остальное нули):")
print(x)
print("one-hot × таблица весов == Embedding:", torch.allclose(x @ model.weight, model(xs[:3])))

logits = model(xs[:1])
print("Сырые оценки для первой пары (сеть ещё не обучена):", logits.detach().round(decimals=2))
print("После softmax — вероятности, их сумма:", torch.softmax(logits, dim=1).sum().item())

# Что делает один шаг обучения? Учим отдельную табличку одной паре «а → .» (имя кончается на «а»)
demo = nn.Embedding(V, V)
a, b = tok.stoi["а"], tok.stoi[BOUNDARY]
demo_opt = torch.optim.SGD(demo.parameters(), lr=1.0)
for i in range(4):
    loss = F.cross_entropy(demo(torch.tensor([a])), torch.tensor([b]))
    demo_opt.zero_grad()
    loss.backward()
    p = torch.softmax(demo.weight[a], dim=0)
    print(f"Шаг {i}: P(«.» после «а») = {p[b]:.3f}, loss = {loss.item():.3f}")
    demo_opt.step()
grad = demo.weight.grad
print("Градиент строки «а» = вероятности − правильный ответ (one-hot):",
      torch.allclose(grad[a], p.detach() - F.one_hot(torch.tensor(b), V)))
print("Строк таблицы, которые изменились бы на этом шаге:", int((grad.abs().sum(1) > 0).sum()), "из", V)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
history = []
for step in range(301):
    loss = F.cross_entropy(model(xs), ys)   # сразу все пары, без мини-батчей: их всего ~1400
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    if step % 50 == 0:
        print(f"Шаг {step:3d}: loss = {loss.item():.3f}")

counts = bigram_counts(names, tok)
probs_counts = (counts + 0.1) / (counts + 0.1).sum(dim=1, keepdim=True)
loss_counts = -probs_counts[xs, ys].log().mean().item()
print(f"Loss нейросети: {history[-1]:.3f}, loss подсчётов из шага 2: {loss_counts:.3f}")

probs_net = torch.softmax(model.weight.detach(), dim=1)
print("После «а»: подсчёты → нейросеть")
for i in probs_counts[a].argsort(descending=True)[:5].tolist():
    print(f"  «{tok.itos[i]}»: {probs_counts[a, i]:.3f} → {probs_net[a, i]:.3f}")

g = torch.Generator().manual_seed(1)
print("Имена от нейросети:")
print(", ".join(sample_name(probs_net, tok, g) for _ in range(20)))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].plot(history)
axes[0].axhline(loss_counts, color="gray", linestyle=":", label="подсчёты (шаг 2)")
axes[0].set_xlabel("шаг обучения")
axes[0].set_ylabel("loss")
axes[0].set_title("Обучение биграммной сети")
axes[0].legend()
axes[0].grid(alpha=0.3)
for ax, p, title in [(axes[1], probs_counts, "Вероятности: подсчёты"), (axes[2], probs_net, "Вероятности: нейросеть")]:
    ax.imshow(p, cmap="Blues")
    ax.set_xticks(range(V), tok.chars, fontsize=7)
    ax.set_yticks(range(V), tok.chars, fontsize=7)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("bigram_neural.png", dpi=100)
show()
