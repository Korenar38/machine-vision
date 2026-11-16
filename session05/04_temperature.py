"""Шаг 4. Температура: как модель выбирает следующий символ.

Модель выдаёт вероятности, а уж как по ним выбирать — решаем мы.
Делим «сырые оценки» на температуру T: маленькая T делает выбор уверенным, большая — случайным.
"""

import matplotlib.pyplot as plt
import torch

from common import apply_temperature, bigram_counts, bigram_pairs, load_names, name_tokenizer, sample_name, show

names = load_names()
tok = name_tokenizer(names)
counts = bigram_counts(names, tok)
logits = torch.log(counts + 0.1)        # «сырые оценки»: softmax(logits) — это вероятности из шага 2

# Сначала на трёх числах: оценки 2, 1, 0 → вероятности при разной температуре
z = torch.tensor([2.0, 1.0, 0.0])
print("Оценки:", z.tolist())
for t in [0, 0.25, 0.5, 1.0, 2.0, 10.0]:
    print(f"  T = {t:>4}: вероятности", ", ".join(f"{p:.3f}" for p in apply_temperature(z, t)))

temperatures = [0.3, 0.7, 1.0, 1.5, 3.0]
letter = "а"
row = logits[tok.stoi[letter]]
fig, axes = plt.subplots(len(temperatures), 1, figsize=(9, 9), sharex=True)
for ax, t in zip(axes, temperatures):
    p = apply_temperature(row, t)
    ax.bar(tok.chars, p)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"T = {t}")
    top = p.argmax().item()
    print(f"T = {t}: самый вероятный после «{letter}» — «{tok.itos[top]}» с вероятностью {p[top]:.2f}")
axes[0].set_title(f"Вероятности следующей буквы после «{letter}» при разной температуре")
plt.tight_layout()
plt.savefig("temperature.png", dpi=100)
show()

g = torch.Generator().manual_seed(1)
for t in [0, 0.5, 1.0, 2.0]:
    probs = apply_temperature(logits, t)
    print(f"T = {t}: {', '.join(sample_name(probs, tok, g) for _ in range(12))}")

# Качество против разнообразия: 500 имён при каждой температуре.
# «Штраф» считаем по честной модели (T = 1): чем он меньше, тем правдоподобнее имена.
base = apply_temperature(logits, 1.0)
print("   T   разных имён из 500   средняя длина   штраф")
for t in [0, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    probs = apply_temperature(logits, t)
    batch = [sample_name(probs, tok, g) for _ in range(500)]
    xs, ys = bigram_pairs(batch, tok)
    penalty = -base[xs, ys].log().mean().item()
    length = sum(map(len, batch)) / len(batch)
    print(f"{t:4}   {len(set(batch)):18d}   {length:13.1f}   {penalty:5.2f}")
