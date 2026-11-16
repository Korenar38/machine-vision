"""Шаг 6. Настоящий GPT: ruGPT-3 small (125 млн весов).

Та же задача, что у мини-GPT из сессии 5, — угадать следующий токен, — но модель в 150 раз больше
и училась на десятках гигабайт русского текста. Смотрим вероятности следующего слова, способы выбора
(жадный, температура, top-k, top-p), сравниваем loss с мини-GPT и подсвечиваем «удивительные» слова.
"""

import importlib.util
import math

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from common import HERE, load_genres, load_model, load_onegin, load_tokenizer, show

tok = load_tokenizer("gpt")
gpt = load_model("gpt", "lm")
print("устройство модели:", gpt.config.n_layer, "блоков,", gpt.config.n_head, "голов, вектор", gpt.config.n_embd)

@torch.no_grad()
def next_token_probs(text):
    ids = tok(text, return_tensors="pt")["input_ids"]
    logits = gpt(ids).logits[0, -1]               # оценки для токена после последнего
    return logits.softmax(-1)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, prompt in zip(axes, ["Мой дядя самых честных", "Жили-были дед да", "Столица России —"]):
    probs = next_token_probs(prompt)
    best = probs.topk(10)
    pieces = [repr(tok.decode([i]))[1:-1].replace(" ", "·") for i in best.indices.tolist()]
    ax.barh(pieces[::-1], best.values.tolist()[::-1], color="tab:orange")
    ax.set_title(f"«{prompt} …»")
    ax.set_xlabel("вероятность")
    print(f"«{prompt}» → " + ", ".join(f"{p} {v:.0%}" for p, v in zip(pieces[:5], best.values.tolist())))
plt.tight_layout()
plt.savefig("gpt_next_token.png", dpi=100)
show()

def continue_text(prompt, max_new_tokens=40, seed=0, **how):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = gpt.generate(ids, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id,
                           do_sample=bool(how), **how)
    return tok.decode(out[0][ids.shape[1]:])


prompt = "Жили-были дед да баба."
for name, how in [("жадный", {}),
                  ("температура 0.7", {"temperature": 0.7, "top_k": 0}),
                  ("температура 1.5", {"temperature": 1.5, "top_k": 0}),
                  ("top-k = 20", {"top_k": 20}),
                  ("top-p = 0.9", {"top_p": 0.9, "top_k": 0})]:
    print(f"\n--- {name}:\n{prompt}{continue_text(prompt, **how)}")

for prompt in ["Новости. Сегодня в Москве", "Рецепт блинов. Возьмите", "Дорогой дневник! Сегодня",
               "Инспектор осмотрел комнату и"]:
    print(f"\n{prompt}{continue_text(prompt, max_new_tokens=35, top_p=0.9, top_k=0, temperature=0.8)}")

start = "Мой дядя самых честных правил,\nКогда не в шутку занемог,\n"
print(start + continue_text(start, max_new_tokens=60))

@torch.no_grad()
def gpt_loss_per_char(text, window=512):
    """Сумма loss всех токенов текста / число символов. Длинный текст режем на куски по window токенов."""
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    total = 0.0
    for i in range(0, len(ids) - 1, window):
        chunk = ids[max(0, i - 1):i + window].unsqueeze(0)          # +1 токен слева: первый токен куска тоже угадываем
        logits = gpt(chunk).logits
        total += F.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum").item()
    return total / len(text)


def load_mini_gpt():
    """Мини-GPT из сессии 5 (session05/gpt.pt) или None, если её ещё не обучили."""
    path = HERE.parent / "session05" / "common.py"
    spec = importlib.util.spec_from_file_location("session05_common", path)
    s5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s5)
    if not (s5.HERE / "gpt.pt").exists():
        return None, None
    return s5.load_gpt("gpt")


@torch.no_grad()
def mini_loss_per_char(model, mini_tok, text, block=64):
    """Средний loss мини-GPT на символ: куски по block символов, как при её обучении."""
    text = "".join(ch for ch in text if ch in mini_tok.stoi)        # символов, которых мини-GPT не знает, выбрасываем
    data = torch.tensor(mini_tok.encode(text))
    total, count = 0.0, 0
    for i in range(0, len(data) - 1, block):
        x = data[i:i + block].unsqueeze(0)
        y = data[i + 1:i + block + 1].unsqueeze(0)
        x = x[:, :y.shape[1]]
        logits = model(x)
        logits = logits[0] if isinstance(logits, tuple) else logits
        total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        count += y.numel()
    return total / count


onegin = load_onegin()
onegin_val = onegin[int(0.9 * len(onegin)):]
new_poems = "\n\n".join(r["text"] for r in load_genres() if r["genre"] == "стихи")
texts = {"«Онегин», последние 10%": onegin_val, "новые стихи из genres.csv": new_poems}

mini, mini_tok = load_mini_gpt()
results = {}
for name, text in texts.items():
    results[name] = {"ruGPT-3": gpt_loss_per_char(text)}
    if mini is not None:
        results[name]["мини-GPT"] = mini_loss_per_char(mini, mini_tok, text)
    print(f"{name:28s} ({len(text)} символов): " + ", ".join(f"{m} {v:.2f}" for m, v in results[name].items()))
if mini is None:
    print("Нет session05/gpt.pt — выполните шаг 6 сессии 5, чтобы сравнить с мини-GPT")
print(f"для сравнения, угадывание наугад среди 75 символов: {math.log(75):.2f}")

@torch.no_grad()
def surprise(text):
    """Список (кусочек текста, удивление) для всех токенов, кроме первого."""
    ids = tok(text, return_tensors="pt")["input_ids"]
    logp = gpt(ids).logits[0, :-1].log_softmax(-1)
    s = -logp[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
    pieces = [tok.decode([i]) for i in ids[0].tolist()]
    return [(pieces[0], 0.0)] + list(zip(pieces[1:], s.tolist()))


def draw_surprise(ax, text):
    """Рисует текст, фон каждого кусочка — по удивлению (0 — белый, 10 и больше — тёмно-красный)."""
    items = surprise(text)
    prev = None
    for piece, s in items:
        style = dict(fontsize=12, va="bottom", color="white" if s > 7 else "black",
                     bbox=dict(boxstyle="square,pad=0.1", facecolor=plt.cm.Reds(min(s / 10, 1.0)), edgecolor="none"))
        if prev is None:
            prev = ax.text(0.01, 0.3, piece, transform=ax.transAxes, **style)
        else:   # следующий кусочек приклеиваем к правому краю предыдущего
            prev = ax.annotate(piece, xy=(1, 0), xycoords=prev, xytext=(1, 0), textcoords="offset points", **style)
    ax.axis("off")
    mean = sum(s for _, s in items[1:]) / max(1, len(items) - 1)
    ax.set_title(f"средний loss {mean:.1f}", fontsize=9, loc="left")


examples = [
    "Мой дядя самых честных правил, когда не в шутку занемог.",
    "Мой дядя самых честных правил, когда не в шутку заболел.",
    "Мой дядя самых честных правил, когда не в шутку купил пылесос.",
    "Мама мыла раму, а папа читал газету.",
    "Мама мыла раму, а папа читал газету в кастрюле.",
    "Вчера я ходил в магазин и купил хлеп и малоко.",
]
fig, axes = plt.subplots(len(examples), 1, figsize=(10, 0.9 * len(examples)))
for ax, text in zip(axes, examples):
    draw_surprise(ax, text)
plt.savefig("gpt_surprise.png", dpi=100)
show()

for text in examples:
    top = sorted(surprise(text)[4:], key=lambda t: -t[1])[:2]      # первые токены пропускаем: там у модели нет контекста
    print(f"{text:65s} самое неожиданное: " + ", ".join(f"«{p.strip()}» {s:.1f}" for p, s in top))
