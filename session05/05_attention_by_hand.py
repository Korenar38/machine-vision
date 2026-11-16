"""Шаг 5. Внимание (attention) своими руками.

Биграмма видит только одну предыдущую букву. Внимание позволяет каждому токену
«посмотреть» на все предыдущие и взять от каждого столько, сколько нужно.
"""

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from common import show

torch.manual_seed(0)
words = ["мой", "дядя", "самых", "честных", "правил"]
T, C = len(words), 8                     # 5 токенов, у каждого вектор из 8 чисел
x = torch.randn(T, C)                    # пока просто случайные «эмбеддинги» слов
print("Вход x:", tuple(x.shape), "— по строке на слово")

# --- 1. Самое простое «внимание»: среднее по себе и всем предыдущим словам
mask = torch.tril(torch.ones(T, T))      # нижний треугольник: смотреть можно только назад
print("Маска (1 — можно смотреть):")
print(mask)
weights_avg = mask / mask.sum(dim=1, keepdim=True)
print("Веса «просто среднее» (каждая строка в сумме = 1):")
print(weights_avg.round(decimals=2))
out_avg = weights_avg @ x
print("Третье слово после усреднения == среднее первых трёх:", torch.allclose(out_avg[2], x[:3].mean(0)))

# --- Внимание на пальцах: один запрос, три ключа, всё можно посчитать в уме
query = torch.tensor([1.0, 0.0])                            # «я ищу что-то про первое число»
keys = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])   # что «предлагает» каждое из трёх слов
values = torch.tensor([[10.0], [20.0], [30.0]])             # что каждое слово отдаст
dots = keys @ query
w = torch.softmax(dots, dim=0)
for i in range(3):
    print(f"слово {i + 1}: Q·K = {dots[i]:.0f}, вес после softmax = {w[i]:.3f}, значение = {values[i, 0]:.0f}")
print(f"Выход = {w[0]:.3f}·10 + {w[1]:.3f}·20 + {w[2]:.3f}·30 = {(w @ values).item():.2f}"
      "  (просто среднее дало бы 20)")

# --- 2. Настоящее внимание: веса зависят от самих слов
# Каждое слово делает три вектора:
#   запрос Q — «что я ищу», ключ K — «что я могу предложить», значение V — «что я отдам».
head = 4
to_q, to_k, to_v = nn.Linear(C, head, bias=False), nn.Linear(C, head, bias=False), nn.Linear(C, head, bias=False)
q, k, v = to_q(x), to_k(x), to_v(x)
scores = q @ k.T / head ** 0.5           # (T, T): насколько запрос строки подходит к ключу столбца
print("Оценки Q·Kᵀ (до маски):")
print(scores.detach().round(decimals=2))
scores = scores.masked_fill(mask == 0, float("-inf"))   # будущее запрещено: −∞ после softmax станет 0
weights = torch.softmax(scores, dim=1)
print("Веса внимания (после маски и softmax):")
print(weights.detach().round(decimals=2))
out = weights @ v
print("Выход:", tuple(out.shape), "— для каждого слова смесь значений V предыдущих слов")

builtin = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)[0]
print("Совпадает со встроенной функцией PyTorch:", torch.allclose(out, builtin, atol=1e-6))

# Зачем делить на √d: чем длиннее векторы, тем больше разброс Q·K и тем «жаднее» softmax
print("   d   разброс Q·K   макс. вес без деления   макс. вес с делением на √d")
for d in [4, 64, 512]:
    qd, kd = torch.randn(1000, 1, d), torch.randn(1000, 5, d)    # 1000 опытов: запрос и 5 ключей
    raw = (qd * kd).sum(-1)                                        # (1000, 5) оценок Q·K
    top_raw = torch.softmax(raw, dim=1).max(1).values.mean()
    top_scaled = torch.softmax(raw / d ** 0.5, dim=1).max(1).values.mean()
    print(f"{d:4d}   {raw.std():11.1f}   {top_raw:21.2f}   {top_scaled:26.2f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, w, title in [(axes[0], weights_avg, "Просто среднее"), (axes[1], weights.detach(), "Внимание Q·K")]:
    ax.imshow(w, cmap="Oranges", vmin=0, vmax=1)
    for i in range(T):
        for j in range(T):
            ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(T), words, rotation=30)
    ax.set_yticks(range(T), words)
    ax.set_xlabel("на какое слово смотрим")
    ax.set_ylabel("кто смотрит")
    ax.set_title(title)
plt.tight_layout()
plt.savefig("attention_weights.png", dpi=100)
show()

# --- 3. Внимание, настроенное вручную: слово ищет похожие на себя слова
# Вектор слова — «какая это часть речи». Пусть запрос = ключ = этот вектор:
# тогда скалярное произведение велико у похожих слов, и они получат большой вес.
x_toy = torch.tensor([
    # местоим. сущ. прилаг.
    [1.0, 0.0, 0.0],   # мой
    [0.0, 1.0, 0.0],   # дядя
    [0.0, 0.0, 1.0],   # самых
    [0.0, 0.0, 1.0],   # честных
    [0.0, 1.0, 0.0],   # правил (здесь считаем его похожим на существительное)
]) * 3
w_toy = torch.softmax((x_toy @ x_toy.T).masked_fill(mask == 0, float("-inf")), dim=1)
print("Веса «ищу похожих»:")
print(w_toy.round(decimals=2))
for i in range(1, T):
    j = w_toy[i, :i].argmax().item()
    if w_toy[i, j] > 0.1:
        print(f"«{words[i]}» из предыдущих слов больше всего смотрит на «{words[j]}» ({w_toy[i, j]:.2f})")
    else:
        print(f"«{words[i]}»: похожих слов раньше не было — смотрит только на себя")


def attend(x):
    """Внимание с маской на слоях to_q, to_k, to_v из части 2."""
    return F.scaled_dot_product_attention(to_q(x)[None], to_k(x)[None], to_v(x)[None], is_causal=True)[0]


# Внимание не знает порядка слов: перемешаем первые четыре — выход последнего слова не изменится
order = [3, 0, 2, 1, 4]
print("Было:  ", " ".join(words))
print("Стало: ", " ".join(words[i] for i in order))
print("Выход для «правил» тот же:", torch.allclose(attend(x)[-1], attend(x[order])[-1], atol=1e-6))
print("Поэтому в GPT (шаг 6) к каждому токену добавляют ещё и эмбеддинг его места — pos_emb")

# --- 4. Несколько голов: несколько независимых «взглядов» сразу
n_head, C_big = 4, 16
x_big = torch.randn(1, T, C_big)
qkv = nn.Linear(C_big, 3 * C_big)(x_big)
q, k, v = qkv.split(C_big, dim=2)
q, k, v = (t.view(1, T, n_head, C_big // n_head).transpose(1, 2) for t in (q, k, v))
print("Q для 4 голов:", tuple(q.shape), "= (пример, головы, токены, числа на голову)")
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("Выход голов:", tuple(out.shape), "→ склеиваем обратно:", tuple(out.transpose(1, 2).reshape(1, T, C_big).shape))
