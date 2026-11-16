"""Шаг 1. Текст — это числа: словарь символов, слова и «склейка» частых пар (BPE).

Нейросеть умеет работать только с числами, поэтому текст сначала режут на кусочки (токены)
и каждому кусочку дают номер.
"""

import re
from collections import Counter

import matplotlib.pyplot as plt

from common import CharTokenizer, load_text, show

text = load_text()
print("Символов в тексте:", len(text))
print("Строк:", text.count("\n"))
print(text[:200])

# --- 1. Токены-символы
tok = CharTokenizer(text)
print("Разных символов (размер словаря):", tok.vocab_size)
print("Словарь:", repr("".join(tok.chars)))

line = "Мой дядя самых честных правил"
ids = tok.encode(line)
print("Строка:", line)
print("Номера:", ids)
print("Обратно:", tok.decode(ids))

# Номер — просто ярлык: у соседних номеров буквы не «похожи», а символа не из словаря нет вовсе
for a, b in [("а", "б"), ("а", "я"), ("а", "А")]:
    print(f"«{a}» = {tok.stoi[a]}, «{b}» = {tok.stoi[b]}: разница номеров {abs(tok.stoi[a] - tok.stoi[b])}")
try:
    tok.encode("Привет, 🙂")
except KeyError as err:
    print("Символа", err, "нет в словаре — модель не сможет его ни прочитать, ни написать")

letters = Counter(ch for ch in text.lower() if ch.isalpha())
top = letters.most_common(33)
plt.figure(figsize=(10, 3.5))
plt.bar([ch for ch, _ in top], [n for _, n in top])
plt.title("Как часто встречаются буквы в «Евгении Онегине»")
plt.ylabel("сколько раз")
plt.tight_layout()
plt.savefig("char_freq.png", dpi=100)
show()

# --- 2. Токены-слова
words = re.findall(r"[а-яё]+", text.lower())
vocab = Counter(words)
print("Слов в тексте:", len(words))
print("Разных слов (размер словаря):", len(vocab))
print("Слов, которые встретились всего один раз:", sum(1 for n in vocab.values() if n == 1))
print("Самые частые слова:", vocab.most_common(10))
for word in ["онегин", "татьяна", "нейросеть"]:
    print(f"Слово «{word}» в словаре: {'да' if word in vocab else 'НЕТ — модель его не знает'}")

# --- 3. Золотая середина: склеиваем частые пары (BPE, как в настоящих GPT)
corpus = {tuple(word): n for word, n in vocab.items()}   # слово → кусочки (сначала по буквам) и частота


def merge(pieces, a, b):
    """Склеить в одном слове все соседние пары (a, b) в один кусочек a + b."""
    out, i = [], 0
    while i < len(pieces):
        if i + 1 < len(pieces) and pieces[i] == a and pieces[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return tuple(out)


def count_pairs(corpus):
    """Сколько раз каждая соседняя пара кусочков встречается во всех словах (с учётом частоты слов)."""
    pairs = Counter()
    for pieces, n in corpus.items():
        for a, b in zip(pieces, pieces[1:]):
            pairs[a, b] += n
    return pairs


# Сначала — игрушечный пример, который легко проверить в уме: три слова и их частоты
toy = {tuple("мама"): 4, tuple("мак"): 3, tuple("ком"): 2}
for step in range(1, 4):
    pairs = count_pairs(toy)
    print(f"Шаг {step}. Пары:", ", ".join(f"«{a}{b}» {n}" for (a, b), n in pairs.most_common()))
    (a, b), n = pairs.most_common(1)[0]
    toy = {merge(pieces, a, b): count for pieces, count in toy.items()}
    print(f"  склеиваем «{a}» + «{b}» → слова теперь:", ", ".join(" · ".join(p) for p in toy))

merges = []
for step in range(1, 201):
    (a, b), n = count_pairs(corpus).most_common(1)[0]
    merges.append((a, b))
    corpus = {merge(pieces, a, b): count for pieces, count in corpus.items()}
    if step <= 20 or step % 50 == 0:
        print(f"Склейка {step:3d}: «{a}» + «{b}» → «{a + b}» (встречалась {n} раз)")


def bpe(word):
    """Разрезать любое слово на кусочки, повторив все склейки по порядку."""
    pieces = tuple(word)
    for a, b in merges:
        pieces = merge(pieces, a, b)
    return pieces


print("Размер словаря BPE: буквы + склейки =", len(set("".join(words))) + len(merges))
for word in ["онегин", "татьяна", "предисловий", "нейросеть"]:
    print(f"{word:>12} → {' | '.join(bpe(word))}")

# Одна и та же строка в трёх токенизациях: чем меньше токенов, тем больше текста влезет в модель
line_words = re.findall(r"[а-яё]+", line.lower())
print(f"«{line}»:")
print("  по символам:", len(tok.encode(line)), "токенов")
print("  по словам:  ", len(line_words), "токенов")
print("  BPE:        ", sum(len(bpe(w)) for w in line_words), "токенов (без пробелов):",
      " · ".join("|".join(bpe(w)) for w in line_words))
