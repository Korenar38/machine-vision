"""Шаг 7. Маленький чат-бот: Qwen2.5-0.5B-Instruct (500 млн весов).

Смотрим, что такое «чат» изнутри (обычный текст со служебными токенами), как системная подсказка
меняет поведение, определяем жанр подсказкой и сравниваем с классификатором из шага 4,
достаём данные из текста в JSON и ловим модель на выдумках (галлюцинациях).
На процессоре один ответ занимает от нескольких секунд до минуты.
"""

import json
import time

import torch

from common import load_genres, load_model, load_tokenizer

tok = load_tokenizer("chat")
model = load_model("chat", "lm")
torch.manual_seed(0)

messages = [
    {"role": "system", "content": "Ты — вежливый помощник школьника. Отвечай кратко."},
    {"role": "user", "content": "Что такое фотосинтез?"},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
print("токенов в подсказке:", len(tok(prompt)["input_ids"]))

def chat(messages, max_new_tokens=200, temperature=None, show_time=True, model=model, tok=tok):
    """Ответ модели на диалог. temperature=None — жадный выбор (одинаковый ответ при каждом запуске)."""
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"]
    if temperature:     # случайный выбор: температура и top-p, как в шаге 6
        how = {"do_sample": True, "temperature": temperature, "top_p": 0.9, "top_k": None}
    else:               # жадный выбор (у Qwen по умолчанию включён случайный — выключаем явно)
        how = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    start = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id, **how)
    answer = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    if show_time:
        n = out.shape[1] - ids.shape[1]
        print(f"[{n} токенов за {time.time() - start:.1f} с]")
    return answer


print(chat(messages))

for question in ["Какого цвета небо?", "Сколько будет 2 + 2?", "Как зовут автора «Евгения Онегина»?"]:
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
        probs = model(**tok(text, return_tensors="pt")).logits[0, -1].softmax(-1)
    best = probs.topk(5)
    print(f"{question:40s} → " + ", ".join(f"{tok.decode([i])!r} {p:.0%}" for p, i in zip(best.values.tolist(), best.indices.tolist())))

question = "Почему зимой холодно?"
for system in ["Ты — учитель физики. Объясни коротко и понятно, для пятиклассника.",
               "Ты — пират. Отвечай как пират, с пиратскими словечками.",
               "Отвечай одним предложением.",
               "Отвечай только словами «да» или «нет»."]:
    print(f"\n--- system: {system}")
    print(chat([{"role": "system", "content": system}, {"role": "user", "content": question}], max_new_tokens=120))

rows = load_genres()
GENRES = list(dict.fromkeys(r["genre"] for r in rows))
test = [r for r in rows if r["split"] == "test"]
STEMS = {"сказ": "сказка", "детект": "детектив", "фантаст": "фантастика", "стих": "стихи", "новост": "новости",
         "научп": "научпоп", "научн": "научпоп"}

INSTRUCTION = ("Определи жанр текста. Ответь одним словом из списка: сказка, детектив, фантастика, стихи, новости, научпоп.\n"
               "(научпоп — научно-популярный текст)")


def parse_genre(answer):
    """Первый жанр, который встречается в ответе модели (или «?», если ни одного)."""
    answer = answer.lower()
    found = [(answer.find(stem), genre) for stem, genre in STEMS.items() if stem in answer]
    return min(found)[1] if found else "?"


def few_shot_messages(text):
    messages = [{"role": "system", "content": INSTRUCTION}]
    for g in GENRES:                                       # по одному обучающему примеру каждого жанра
        example = next(r["text"] for r in rows if r["genre"] == g and r["split"] == "train")
        messages += [{"role": "user", "content": example}, {"role": "assistant", "content": g}]
    return messages + [{"role": "user", "content": text}]


def zero_shot_messages(text):
    return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": text}]


def genre_test(make, name, **chat_args):
    """Точность чат-бота на 30 проверочных отрывках + какие ответы он давал."""
    start = time.time()
    answers = [parse_genre(chat(make(r["text"]), max_new_tokens=6, show_time=False, **chat_args)) for r in test]
    acc = sum(a == r["genre"] for a, r in zip(answers, test)) / len(test)
    counts = {g: answers.count(g) for g in GENRES + ["?"] if answers.count(g)}
    print(f"{name}: точность {acc:.0%} ({time.time() - start:.0f} с на {len(test)} отрывков), ответы: {counts}")
    return acc


results = {"zero-shot": genre_test(zero_shot_messages, "zero-shot"),
           "few-shot": genre_test(few_shot_messages, "few-shot")}
print("для сравнения, голова из шага 4: 93%, наугад: 17%")

from common import is_downloaded

BIG_MODEL = None            # None — только если уже скачана; True — скачать (~3 ГБ); False — пропустить
if BIG_MODEL or (BIG_MODEL is None and is_downloaded("chat_big")):
    big_tok = load_tokenizer("chat_big")
    big_model = load_model("chat_big", "lm")
    results["1.5B zero-shot"] = genre_test(zero_shot_messages, "Qwen 1.5B, zero-shot", model=big_model, tok=big_tok)
    for question in ["Кто написал роман «Евгений Онегин»?", "Расскажи кратко о романе Пушкина «Капитан Немо»."]:
        print(f"\n>>> {question}\n{chat(question, max_new_tokens=80, model=big_model, tok=big_tok)}")
    del big_model
else:
    print("Пропускаем: модели Qwen2.5-1.5B нет в data/hf. Поставьте BIG_MODEL = True, чтобы скачать её (~3 ГБ).")

ad = ("Внимание! В субботу, 14 декабря, в 11:00 в актовом зале школы № 5 пройдёт олимпиада по информатике "
      "для учеников 7–9 классов. Регистрация у Марии Ивановны в кабинете 203 до пятницы.")
answer = chat([
    {"role": "system", "content": "Извлеки из текста данные и ответь ТОЛЬКО JSON с ключами: событие, дата, время, место, кто_участвует, контакт."},
    {"role": "user", "content": ad},
], max_new_tokens=150)
print(answer)
try:
    data = json.loads(answer.strip("`").removeprefix("json").strip())
    print("\nJSON прочитан программой, например, дата:", data.get("дата"))
except json.JSONDecodeError:
    print("\nМодель ответила не чистым JSON — программа не может это прочитать. Так тоже бывает!")

for question in [
    "Кто написал роман «Евгений Онегин»?",
    "Расскажи кратко о романе Пушкина «Капитан Немо».",            # такого романа нет
    "Кто такой знаменитый русский писатель Аркадий Пупырышкин?",   # и такого писателя нет
    "Сколько будет 37 × 43?",                                       # правильно: 1591
    "Сколько букв «р» в слове «программирование»?",                 # правильно: 3
]:
    print(f"\n>>> {question}\n{chat(question, max_new_tokens=100)}")

dialog = [{"role": "system", "content": "Ты — дружелюбный помощник. Отвечай кратко, по-русски."}]
for question in ["Меня зовут Маша. Я люблю астрономию.", "Посоветуй, что почитать по моему увлечению.", "Как меня зовут?"]:
    dialog.append({"role": "user", "content": question})
    answer = chat(dialog, max_new_tokens=120)
    dialog.append({"role": "assistant", "content": answer})
    print(f"\n>>> {question}\n{answer}")
print("\nсообщений в диалоге:", len(dialog), "— всё это модель читает заново при каждом ответе")
