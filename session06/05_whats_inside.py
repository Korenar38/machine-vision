"""Шаг 5. Что видит BERT: угадывание пропущенного слова, предвзятость, слово в контексте.

BERT учили угадывать закрытые слова в тексте. Смотрим, что он угадывает хорошо, где ошибается,
какие стереотипы выучил из текстов и как одно и то же слово получает разные векторы в разных фразах.
"""

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from common import load_model, load_tokenizer, show

tok = load_tokenizer("bert")
mlm = load_model("bert", "mlm")


@torch.no_grad()
def mask_probs(text):
    """Вероятности всех токенов словаря на месте [MASK]."""
    batch = tok(text, return_tensors="pt")
    pos = (batch["input_ids"][0] == tok.mask_token_id).nonzero()[0, 0]
    return mlm(**batch).logits[0, pos].softmax(-1)


def guess(text, k=6):
    probs = mask_probs(text)
    best = probs.topk(k)
    words = [f"{tok.convert_ids_to_tokens([i])[0]} {p:.0%}" for p, i in zip(best.values.tolist(), best.indices.tolist())]
    print(f"{text:45s} → {', '.join(words)}")

for text in ["Москва — столица [MASK].", "Я пью чай с [MASK].", "На небе светит яркое [MASK].",
             "Летом мы поехали на [MASK] купаться.", "Мой дядя самых честных [MASK].",
             "Пушкин написал роман «Евгений [MASK]».", "Дважды два равно [MASK].",
             "Учитель поставил мне [MASK] за контрольную."]:
    guess(text)

he, she = tok.convert_tokens_to_ids("Он"), tok.convert_tokens_to_ids("Она")
professions = ["медсестрой", "воспитателем", "врачом", "парикмахером", "учителем", "бухгалтером",
               "секретарём", "программистом", "строителем", "пилотом", "водителем", "инженером"]
share = {}
for prof in professions:
    p = mask_probs(f"[MASK] работает {prof}.")
    share[prof] = (p[she] / (p[he] + p[she])).item()
    print(f"[MASK] работает {prof:14s} P(Он) = {p[he]:.3f}, P(Она) = {p[she]:.3f}, доля «Она» {share[prof]:.0%}")

order = sorted(share, key=share.get)
plt.figure(figsize=(8, 5))
plt.barh(order, [share[p] for p in order], color=["tab:orange" if share[p] > 0.5 else "tab:blue" for p in order])
plt.axvline(0.5, color="gray", linestyle=":")
plt.xlim(0, 1)
plt.xlabel("доля «Она» среди «Он»/«Она»")
plt.title("«[MASK] работает ...»: кого модель представляет в профессии")
plt.tight_layout()
plt.savefig("bias.png", dpi=100)
show()

gpt_tok = load_tokenizer("gpt")
gpt = load_model("gpt")

words = {
    "ключ": ["Я открыл дверь и положил в карман ключ.", "Он вставил в замок ключ.", "Мама дала мне от квартиры ключ.",
             "Из-под камня бил холодный ключ.", "В лесу из земли бьёт чистый ключ.", "В овраге журчал родниковый ключ."],
    "лук": ["Бабушка добавила в суп лук.", "Мама порезала для салата лук.", "На грядке вырос зелёный лук.",
            "Охотник натянул лук.", "Лучник взял стрелы и лук.", "Робин Гуд натянул тугой лук."],
    "замок": ["Я закрыл дверь на замок.", "На сейфе сломался замок.", "На двери висел тяжёлый амбарный замок.",
              "Король жил в старинном замке.", "На горе стоит рыцарский замок.", "Над рвом возвышался каменный замок."],
}


@torch.no_grad()
def word_vectors(sentence, word):
    """Векторы слова после каждого слоя: список из 13 векторов (эмбеддинг + 12 блоков)."""
    stem = word[:3]                                       # «замке» тоже найдётся по «зам»; берём последнее вхождение
    start = sentence.rindex(stem)
    end = start + len(stem)
    batch = gpt_tok(sentence, return_tensors="pt", return_offsets_mapping=True)
    offsets = batch.pop("offset_mapping")[0].tolist()
    positions = [i for i, (s, e) in enumerate(offsets) if s < end and e > start]   # токены, задевающие слово
    layers = gpt(**batch, output_hidden_states=True).hidden_states
    return [F.normalize(h[0, positions].mean(0), dim=0) for h in layers]


vectors = {w: [word_vectors(s, w) for s in sentences] for w, sentences in words.items()}

# Близость внутри одного значения и между разными значениями — для каждого слоя
plt.figure(figsize=(8, 4.5))
for w, color in zip(words, ["tab:blue", "tab:green", "tab:orange"]):
    gaps = []
    for layer in range(13):
        M = torch.stack([v[layer] for v in vectors[w]])
        S = M @ M.T
        same = ((S[:3, :3].sum() - 3) / 6 + (S[3:, 3:].sum() - 3) / 6) / 2     # пары с одним значением
        different = S[:3, 3:].mean()                                         # пары с разными значениями
        gaps.append((same - different).item())
    plt.plot(range(13), gaps, marker="o", color=color, label=f"«{w}»")
plt.xlabel("слой (0 — эмбеддинг, 12 — последний блок)")
plt.ylabel("близость «одно значение» − «разные значения»")
plt.title("Слой за слоем модель разводит значения слова")
plt.legend()
plt.tight_layout()
plt.savefig("word_layers.png", dpi=100)
show()

# Подробно: все попарные близости «лука» и «ключа» на последнем слое
examples = [(s, w) for w in ["ключ", "лук"] for s in words[w]]
V = torch.stack([vectors[w][words[w].index(s)][-1] for s, w in examples])
sim = V @ V.T
plt.figure(figsize=(8, 6.5))
plt.imshow(sim, cmap="viridis", vmin=0.4, vmax=1)
plt.colorbar(label="близость векторов слова")
labels = [s[:32] for s, _ in examples]
plt.xticks(range(len(examples)), labels, rotation=60, ha="right", fontsize=8)
plt.yticks(range(len(examples)), labels, fontsize=8)
for i in range(len(examples)):
    for j in range(len(examples)):
        plt.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=6, color="black" if sim[i, j] > 0.8 else "white")
plt.title("Векторы слов «ключ» и «лук» на последнем слое")
plt.tight_layout()
plt.savefig("word_in_context.png", dpi=100)
show()
