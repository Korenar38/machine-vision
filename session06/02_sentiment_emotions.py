"""Шаг 2. Тон и эмоции текста: готовые классификаторы.

Две готовые модели на основе rubert-tiny2: одна определяет тон (позитив / нейтрально / негатив),
другая — эмоции (радость, грусть, удивление, страх, злость). Ищем фразы, на которых они ошибаются,
и строим «эмоциональную кардиограмму» «Евгения Онегина».
"""

import matplotlib.pyplot as plt
import torch

from common import load_model, load_tokenizer, model_dir, onegin_stanzas, load_onegin, show
from transformers import pipeline     # после common: там отключены лишние предупреждения

classify = pipeline("text-classification", model=str(model_dir("sentiment")), top_k=None)

RU = {"neutral": "нейтрально", "positive": "позитив", "negative": "негатив",
      "no_emotion": "без эмоций", "joy": "радость", "sadness": "грусть", "surprise": "удивление",
      "fear": "страх", "anger": "злость"}

for text in ["Отличный фильм, всем советую!", "Ужасное обслуживание, больше не приду.", "Магазин работает с 9 до 21."]:
    result = classify(text)[0]                 # список {"label", "score"} от самого вероятного
    print(f"{text:40s} → " + ", ".join(f"{RU[r['label']]} {r['score']:.0%}" for r in result))

tok = load_tokenizer("sentiment")
model = load_model("sentiment", "classifier")
print("голова модели:", model.classifier)

text = "Отличный фильм, всем советую!"
batch = tok(text, return_tensors="pt")
print("токены:", tok.convert_ids_to_tokens(batch["input_ids"][0]))
with torch.no_grad():
    hidden = model.bert(**batch).last_hidden_state      # (1, число токенов, 312)
    cls = hidden[:, 0]                                  # вектор токена [CLS], (1, 312)
    pooled = torch.tanh(model.bert.pooler.dense(cls))   # pooler: Linear + tanh
    logits = model.classifier(pooled)                   # голова, (1, 3)
probs = torch.softmax(logits, dim=-1)[0]
print("форма векторов после BERT:", tuple(hidden.shape))
print("логиты:", [round(x, 2) for x in logits[0].tolist()])
print("вероятности:", {RU[model.config.id2label[i]]: f"{p:.0%}" for i, p in enumerate(probs.tolist())})

tricky = [
    "Фильм хороший.",
    "Фильм не плохой.",                                   # отрицание
    "Не сказать, что фильм хороший.",                     # отрицание хорошего
    "Ну просто супер: опять дождь и опять пробки.",       # сарказм
    "Начало было скучным, но финал потрясающий!",         # смена тона
    "Ваще огонь 🔥🔥🔥",                                   # сленг и смайлики
    "Отлчный фльм, всем саветую",                         # опечатки
]


def sentiment_table(texts):
    for text in texts:
        result = {r["label"]: r["score"] for r in classify(text)[0]}
        best = max(result, key=result.get)
        print(f"{text:48s} → {RU[best]:10s} " + "  ".join(f"{RU[k][:3]} {result[k]:4.0%}" for k in ["positive", "neutral", "negative"]))


sentiment_table(tricky)

emotions = pipeline("text-classification", model=str(model_dir("emotion")), top_k=None)
EMO = ["joy", "sadness", "surprise", "fear", "anger", "no_emotion"]
COLORS = ["gold", "tab:blue", "tab:purple", "tab:gray", "tab:red", "lightgray"]

phrases = [
    "Какое счастье — наконец каникулы!",
    "Мне грустно, что лето закончилось.",
    "Неужели это правда? Не может быть!",
    "Мне очень страшно, я боюсь темноты.",
    "Как же меня бесит этот шум!",
    "Я рад, что мы победили, но жаль, что лето кончилось.",
    "Поезд отправляется в 10:15.",
    "Ура, мы выиграли олимпиаду!",
    "Сколько можно опаздывать?! Я жду уже час!",
]
table = []
for text in phrases:
    scores = {r["label"]: r["score"] for r in emotions(text)[0]}
    table.append([scores[e] for e in EMO])
    print(f"{text:55s} " + " ".join(f"{RU[e][:5]} {scores[e]:.2f}" for e in EMO if scores[e] > 0.15))

table = torch.tensor(table)
fig, ax = plt.subplots(figsize=(10, 5))
for j, (e, c) in enumerate(zip(EMO, COLORS)):
    ax.barh([i + (j - 2.5) * 0.13 for i in range(len(phrases))], table[:, j], height=0.13, color=c, label=RU[e])
ax.set_yticks(range(len(phrases)), [p if len(p) < 40 else p[:38] + "…" for p in phrases])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("вероятность (у каждой эмоции своя, в сумме не обязательно 1)")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Эмоции в фразах")
plt.tight_layout()
plt.savefig("emotions.png", dpi=100)
show()

text = load_onegin()
stanzas, chapter_starts = [], []
for part in text.split("ГЛАВА")[1:]:                 # роман разбит на главы словом «ГЛАВА»
    chapter_starts.append(len(stanzas))
    stanzas += onegin_stanzas(part)
print(f"Строф: {len(stanzas)}, глав: {len(chapter_starts)}")

scores = []
for i in range(0, len(stanzas), 32):
    for result in emotions(stanzas[i:i + 32], truncation=True):
        s = {r["label"]: r["score"] for r in result}
        scores.append([s["joy"], s["sadness"]])
scores = torch.tensor(scores)


def smooth(x, k=7):
    """Скользящее среднее: каждая точка — среднее k соседних."""
    return torch.nn.functional.avg_pool1d(x.view(1, 1, -1), k, stride=1, padding=k // 2, count_include_pad=False).view(-1)


plt.figure(figsize=(12, 4))
plt.plot(smooth(scores[:, 0]), color="gold", label="радость")
plt.plot(smooth(scores[:, 1]), color="tab:blue", label="грусть")
for c, start in enumerate(chapter_starts):
    plt.axvline(start, color="gray", linestyle=":")
    plt.text(start + 1, plt.ylim()[1] * 0.95, f"гл. {c + 1}", fontsize=8, color="gray")
plt.xlabel("номер строфы")
plt.ylabel("вероятность (среднее по 7 строфам)")
plt.title("Радость и грусть по строфам «Евгения Онегина»")
plt.legend()
plt.tight_layout()
plt.savefig("onegin_emotions.png", dpi=100)
show()

for label, col in [("радостная", 0), ("грустная", 1)]:
    i = scores[:, col].argmax().item()
    print(f"\nСамая {label} строфа по мнению модели (№ {i}, {scores[i, col]:.0%}):\n{stanzas[i]}")
