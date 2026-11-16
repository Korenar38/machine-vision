"""Шаг 2. Самая простая языковая модель: биграммы по подсчётам.

Считаем, какая буква за какой идёт в русских именах, и придумываем новые имена по этой таблице.
"""

import matplotlib.pyplot as plt
import torch

from common import BOUNDARY, bigram_counts, bigram_pairs, load_names, name_tokenizer, sample_name, show

names = load_names()
print("Имён:", len(names), "— например:", names[:8])
tok = name_tokenizer(names)
print("Символов в словаре (точка + буквы):", tok.vocab_size, repr("".join(tok.chars)))

xs, ys = bigram_pairs(names, tok)
print("Пар «буква → следующая»:", len(xs))
print("Пары из имени «анна»:", [tok.decode([x]) + tok.decode([y]) for x, y in zip(*bigram_pairs(["анна"], tok))])

counts = bigram_counts(names, tok)
print("Чаще всего имя начинается на:", [tok.itos[i] for i in counts[0].argsort(descending=True)[:5].tolist()])
print("Чаще всего имя заканчивается на:", [tok.itos[i] for i in counts[:, 0].argsort(descending=True)[:5].tolist()])

plt.figure(figsize=(9, 8))
plt.imshow(counts, cmap="Blues")
plt.xticks(range(tok.vocab_size), tok.chars)
plt.yticks(range(tok.vocab_size), tok.chars)
plt.xlabel("следующая буква")
plt.ylabel("текущая буква")
plt.title("Сколько раз за буквой (строка) идёт буква (столбец)")
plt.colorbar(shrink=0.8)
plt.tight_layout()
plt.savefig("bigram_counts.png", dpi=100)
show()

# Подсчёты → вероятности: каждую строку делим на её сумму.
# +0.1 ко всем клеткам («сглаживание»): иначе невиданная пара получит вероятность ровно 0.
probs = (counts + 0.1) / (counts + 0.1).sum(dim=1, keepdim=True)
print("После «а» чаще всего идут:",
      [(tok.itos[i], round(probs[tok.stoi['а'], i].item(), 3)) for i in probs[tok.stoi["а"]].argsort(descending=True)[:5].tolist()])

# «Нечестный кубик»: 1000 раз выбираем первую букву имени и сравниваем частоты с вероятностями
g = torch.Generator().manual_seed(0)
rolls = torch.multinomial(probs[0], 1000, replacement=True, generator=g)
print("Первая буква: вероятность в таблице → доля из 1000 бросков")
for i in probs[0].argsort(descending=True)[:5].tolist():
    print(f"  «{tok.itos[i]}»: {probs[0, i]:.3f} → {(rolls == i).float().mean():.3f}")

g = torch.Generator().manual_seed(1)
print("Придуманные имена:")
print(", ".join(sample_name(probs, tok, g) for _ in range(20)))

uniform = torch.ones_like(probs) / tok.vocab_size
print("А это буквы наугад, без всякой модели:")
print(", ".join(sample_name(uniform, tok, g) for _ in range(5)))

# Вероятность целого имени = произведение вероятностей всех его пар
ids = tok.encode(BOUNDARY + "анна" + BOUNDARY)
p_name = 1.0
for a, b in zip(ids, ids[1:]):
    p_name *= probs[a, b].item()
    print(f"P(«{tok.itos[b]}» после «{tok.itos[a]}») = {probs[a, b]:.3f}   произведение пока: {p_name:.6f}")
print("Произведения быстро становятся крошечными, поэтому вместо них складывают логарифмы:")
print(f"  log(произведение) = {torch.log(torch.tensor(p_name)):.3f}, "
      f"сумма логарифмов = {sum(probs[a, b].log().item() for a, b in zip(ids, ids[1:])):.3f}")


def name_loss(name):
    """Средний штраф модели на одном имени: чем меньше, тем «правдоподобнее» имя для биграммы."""
    x, y = bigram_pairs([name], tok)
    return -probs[x, y].log().mean().item()


for name in ["анна", "виталий", "ааааа", "йьчх"]:
    print(f"{name:>8}: штраф {name_loss(name):.2f} ({'есть' if name in names else 'нет'} в списке имён)")

# Насколько модель хороша? Средний «штраф» −log(вероятность правильной следующей буквы).
# Это та же cross-entropy, что в сессиях 3–4: чем меньше, тем лучше модель угадывает.
loss = -probs[xs, ys].log().mean()
print(f"Средний штраф (loss) биграммной модели: {loss:.3f}")
print(f"Средний штраф угадывания наугад: {torch.log(torch.tensor(tok.vocab_size)):.3f}")
