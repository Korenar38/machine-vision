"""Шаг 6. Мини-GPT учится писать «как Пушкин».

Модель по символам: 4 блока трансформера, внимание из шага 5, температура из шага 4.
Учится угадывать следующий символ «Евгения Онегина». Сохраняет gpt.pt и gpt.json.
"""

import math
import time

import matplotlib.pyplot as plt
import torch

from common import CharTokenizer, MiniGPT, count_params, load_text, save_gpt, show

torch.manual_seed(0)
text = load_text()
tok = CharTokenizer(text)
data = torch.tensor(tok.encode(text))
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]          # последние 10% текста модель не видит при обучении
print(f"Символов: {len(data)}, для обучения {len(train_data)}, для проверки {len(val_data)}")

BLOCK = 64          # сколько символов модель видит за раз
BATCH = 32          # сколько кусков текста в одном шаге обучения
STEPS = 3000        # шагов обучения (на обычном ноутбуке — несколько минут)


def get_batch(source):
    """Случайные куски текста x и те же куски, сдвинутые на один символ вперёд, y."""
    starts = torch.randint(len(source) - BLOCK - 1, (BATCH,))
    x = torch.stack([source[s:s + BLOCK] for s in starts])
    y = torch.stack([source[s + 1:s + BLOCK + 1] for s in starts])
    return x, y


x, y = get_batch(train_data)
print("Пример входа: ", repr(tok.decode(x[0][:30])))
print("Что угадывать:", repr(tok.decode(y[0][:30])))

model = MiniGPT(tok.vocab_size, block_size=BLOCK)
print("Весов в модели:", count_params(model))
for part in ["token_emb", "pos_emb", "blocks", "head"]:
    print(f"  {part:>9}: {count_params(getattr(model, part)):7d}")

# Путь одного батча через модель (dropout выключен, чтобы не тратить случайные числа)
model.eval()
with torch.no_grad():
    emb = model.token_emb(x) + model.pos_emb(torch.arange(BLOCK))
    print("Вход x:", tuple(x.shape), "= (кусков, символов в куске)")
    print("Эмбеддинги символов + мест:", tuple(emb.shape), "= (кусков, символов, чисел на символ)")
    print("После 4 блоков:", tuple(model.blocks(emb).shape), "— форма та же, меняется содержание")
    logits, loss = model(x, y)
    print("Оценки следующего символа:", tuple(logits.shape), f"= (кусков, символов, {tok.vocab_size} вариантов)")
model.train()
print(f"Loss до обучения: {loss:.3f}, угадывание наугад: log({tok.vocab_size}) = {math.log(tok.vocab_size):.3f}")


@torch.no_grad()
def estimate_loss(iters=20):
    """Средний loss на обучающем и проверочном тексте."""
    model.eval()
    result = {}
    for name, source in [("train", train_data), ("val", val_data)]:
        result[name] = torch.stack([model(*get_batch(source))[1] for _ in range(iters)]).mean().item()
    model.train()
    return result


def sample(prompt="\n", n_new=200, temperature=0.8, seed=0):
    """Дописать текст к началу prompt."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt)])
    out = tok.decode(model.generate(idx, n_new, temperature, g)[0])
    model.train()
    return out


history = {"step": [], "train": [], "val": [], "samples": []}
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
start = time.time()
for step in range(STEPS + 1):
    if step % 250 == 0:
        losses = estimate_loss()
        text_now = sample("Мой дядя", n_new=120)
        for key in ("train", "val"):
            history[key].append(losses[key])
        history["step"].append(step)
        history["samples"].append(text_now)
        print(f"\nШаг {step:4d} ({time.time() - start:4.0f} с): loss обуч. {losses['train']:.3f}, проверка {losses['val']:.3f}")
        print(text_now)
    if step == STEPS:
        break
    x, y = get_batch(train_data)
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

save_gpt(model, tok, "gpt", history)

plt.figure(figsize=(7, 3.8))
plt.plot(history["step"], history["train"], "o-", label="обучающий текст")
plt.plot(history["step"], history["val"], "o-", label="проверочный текст")
plt.xlabel("шаг обучения")
plt.ylabel("loss")
plt.title("Мини-GPT учится на «Евгении Онегине»")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("gpt_curves.png", dpi=100)
show()


def visible(s):
    """Строка с видимым переводом строки (↵) — чтобы его было видно при печати."""
    return s.replace("\n", "↵")


@torch.no_grad()
def next_char(prompt, k=5):
    """k самых вероятных следующих символов после prompt по мнению модели."""
    model.eval()
    logits, _ = model(torch.tensor([tok.encode(prompt)]))
    model.train()
    p, idx = torch.softmax(logits[0, -1], dim=0).topk(k)
    return ", ".join(f"«{visible(tok.itos[i])}» {pi:.2f}" for pi, i in zip(p.tolist(), idx.tolist()))


print("Что модель думает о следующем символе:")
for prompt in ["Мой дядя самых честных прави", "Онеги", "Татьян", "Мой дядя самых честных правил,\n", "Евгений"]:
    print(f"  «{visible(prompt)}» → {next_char(prompt)}")

# Куда смотрят головы внимания последнего блока, когда модель читает строку
line = "Мой дядя самых честных правил"
model.eval()
with torch.no_grad():
    model(torch.tensor([tok.encode(line)]))
model.train()
weights = model.blocks[-1].attn.last_weights[0]          # (головы, символы, символы)
fig, axes = plt.subplots(1, weights.shape[0], figsize=(16, 4.6))
for h, ax in enumerate(axes):
    ax.imshow(weights[h], cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(line)), list(line), fontsize=7)
    ax.set_yticks(range(len(line)), list(line), fontsize=7)
    ax.set_title(f"голова {h + 1}")
axes[0].set_ylabel("кто смотрит")
fig.suptitle("Веса внимания обученной модели (последний блок): строка — кто смотрит, столбец — на кого")
plt.tight_layout()
plt.savefig("gpt_attention.png", dpi=100)
show()

print("\n=== Допиши за Пушкина ===")
for prompt in ["Онегин, добрый мой приятель,\n", "Татьяна", "Зима!"]:
    for t in [0.5, 1.0]:
        print(f"\n--- «{prompt.strip()}», температура {t}:")
        print(sample(prompt, n_new=250, temperature=t, seed=1))


def longest_copy(generated):
    """Самый длинный кусок generated, который дословно есть в «Онегине»."""
    best = ""
    for i in range(len(generated)):
        while i + len(best) < len(generated) and generated[i:i + len(best) + 1] in text:
            best = generated[i:i + len(best) + 1]
    return best


# Пишет ли модель своё или списывает? Ищем самый длинный дословный кусок из романа
for t in [0.5, 1.0]:
    generated = sample("\n", n_new=500, temperature=t, seed=2)
    copy = longest_copy(generated)
    print(f"\nТемпература {t}: самый длинный кусок из «Онегина» — {len(copy)} символов: «{visible(copy)}»")
