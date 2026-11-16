"""Шаг 1. Готовые модели: откуда они берутся и как режут текст.

Скачиваем три настоящие языковые модели с Hugging Face Hub, считаем их веса
и сравниваем с мини-GPT из сессии 5. Смотрим, как каждая режет одну и ту же строку на токены.
При первом запуске модели скачиваются в data/hf (около 1,7 ГБ, нужен интернет).
"""

import matplotlib.pyplot as plt

from common import MODELS, count_params, load_model, load_tokenizer, show

MINI_GPT = 820_000         # весов в мини-GPT из сессии 5 (шаг 6)

sizes = {"мини-GPT\n(сессия 5)": MINI_GPT}
for name, kind in [("bert", "base"), ("gpt", "lm"), ("chat", "lm")]:
    model = load_model(name, kind)
    n = count_params(model)
    sizes[f"{name}\n{MODELS[name].split('/')[1]}"] = n
    print(f"{MODELS[name]:45s} {n / 1e6:7.1f} млн весов, в {n / MINI_GPT:4.0f} раз больше мини-GPT")
    del model

plt.figure(figsize=(8, 4))
names = list(sizes)
plt.bar(names, [sizes[n] for n in names], color=["tab:gray", "tab:blue", "tab:orange", "tab:green"])
for i, n in enumerate(names):
    plt.text(i, sizes[n] * 1.2, f"{sizes[n] / 1e6:.1f} млн", ha="center")
plt.yscale("log")
plt.ylim(1e5, 3e9)
plt.ylabel("весов (логарифмическая шкала)")
plt.title("Наша мини-GPT и готовые модели")
plt.tight_layout()
plt.savefig("model_sizes.png", dpi=100)
show()

tokenizers = {name: load_tokenizer(name) for name in ["bert", "gpt", "chat"]}


def pieces(name, text):
    """Текст → список кусочков-токенов (служебные токены [CLS]/[SEP] не показываем)."""
    tok = tokenizers[name]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if name == "bert":
        return tok.convert_ids_to_tokens(ids)            # у BERT кусочки и так читаемые
    return [tok.decode([i]) for i in ids]                # у GPT и Qwen собираем кусочек из байтов


def show_pieces(text):
    print(f"«{text}» — {len(text)} символов")
    for name in tokenizers:
        p = pieces(name, text)
        print(f"  {name:5s} {len(p):3d} токенов: " + " ".join(f"[{x.replace(' ', '·')}]" for x in p))


show_pieces("Мой дядя самых честных правил")
show_pieces("My uncle has most honest principles")

samples = {
    "Онегин": "Мой дядя самых честных правил, когда не в шутку занемог, он уважать себя заставил и лучше выдумать не мог.",
    "новость": "Центральный банк сохранил ключевую ставку на прежнем уровне. Решение объявлено в пятницу.",
    "английский": "The Central Bank kept the key interest rate unchanged. The decision was announced on Friday.",
    "код": "for i in range(10):\n    print(i * i)",
    "смайлики": "Отлично 😀👍🔥",
}
print(f"{'текст':12s} {'символов':>9s} " + " ".join(f"{n:>6s}" for n in tokenizers))
for label, text in samples.items():
    counts = [len(tokenizers[n](text, add_special_tokens=False)["input_ids"]) for n in tokenizers]
    print(f"{label:12s} {len(text):9d} " + " ".join(f"{c:6d}" for c in counts))

print("BERT видит смайлики как:", pieces("bert", "Отлично 😀 😡"))
print("Qwen видит смайлик как:  ", pieces("chat", "😀"), "— один токен")
print("Qwen видит иероглиф 𓀀 как:", pieces("chat", "𓀀"), "— три кусочка байтов")

for name, tok in tokenizers.items():
    print(f"{name:5s} словарь: {len(tok):7d} токенов")
print("в сессии 5:   75 символов")

ids = tokenizers["bert"]("Привет, мир!")["input_ids"]
print("\nBERT добавляет служебные токены:", tokenizers["bert"].convert_ids_to_tokens(ids))
print("их номера:", ids)
print("служебные токены Qwen:", tokenizers["chat"].all_special_tokens[:6])

for word in ["кот", "котёнок", "нейросеть", "Онегин", "автомобилестроение", "пупырышкин"]:
    print(f"{word:20s} BERT: {pieces('bert', word)}   GPT: {pieces('gpt', ' ' + word)}")
