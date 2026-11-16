"""Общие части сессии 5: тексты, словарь символов, биграммы и маленький GPT."""

import importlib.util
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DATA = Path(__file__).parent.parent / "data"
HERE = Path(__file__).parent

BOUNDARY = "."   # служебный символ «начало / конец имени» для биграмм


def load_names():
    """Список русских имён из names.txt (по одному на строку, маленькими буквами)."""
    return (HERE / "names.txt").read_text(encoding="utf-8").split()


def load_text():
    """Весь «Евгений Онегин» (главы 1–8) одной строкой."""
    return (HERE / "onegin.txt").read_text(encoding="utf-8")


class CharTokenizer:
    """Словарь символов: каждый символ текста получает свой номер.

    encode — строка → список номеров, decode — список номеров → строка.
    """

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # символ → номер
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   # номер → символ

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def name_tokenizer(names):
    """Словарь для имён: служебная точка (номер 0) + все буквы, что встречаются в именах."""
    return CharTokenizer(BOUNDARY + "".join(names))


def bigram_pairs(names, tok):
    """Все пары «буква → следующая буква» из имён: два тензора номеров (x, y).

    К каждому имени по краям добавляется точка: «.анна.» → (.,а) (а,н) (н,н) (н,а) (а,.).
    """
    xs, ys = [], []
    for name in names:
        ids = tok.encode(BOUNDARY + name + BOUNDARY)
        xs += ids[:-1]
        ys += ids[1:]
    return torch.tensor(xs), torch.tensor(ys)


def bigram_counts(names, tok):
    """Таблица N[i, j] — сколько раз за символом i шёл символ j."""
    xs, ys = bigram_pairs(names, tok)
    counts = torch.zeros(tok.vocab_size, tok.vocab_size)
    counts.index_put_((xs, ys), torch.ones(len(xs)), accumulate=True)
    return counts


def sample_name(probs, tok, generator=None, max_len=20):
    """Придумать имя по таблице вероятностей probs[i, j] = P(следующий j | текущий i)."""
    i, out = 0, []                     # начинаем с точки (номер 0)
    while len(out) < max_len:
        i = torch.multinomial(probs[i], 1, generator=generator).item()
        if i == 0:                      # снова точка — имя закончилось
            break
        out.append(i)
    return tok.decode(out)


def apply_temperature(logits, temperature):
    """Вероятности из «сырых оценок» logits с температурой.

    temperature < 1 — уверенный и однообразный выбор, > 1 — смелый и случайный.
    temperature = 0 — всегда самый вероятный вариант (жадный выбор).
    """
    if temperature == 0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits / temperature, dim=-1)


# ---------------------------------------------------------------- маленький GPT

class SelfAttention(nn.Module):
    """Многоголовое внимание «только назад»: каждый символ смотрит на себя и предыдущие.

    Это ровно то, что шаг 5 считает вручную, только для нескольких голов сразу.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)       # запрос (Q), ключ (K), значение (V) одним слоем
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        # нижнетреугольная маска: 1 — можно смотреть, 0 — это «будущее», смотреть нельзя
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.last_weights = None                        # веса внимания последнего прохода (для картинок)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, головы, T, C / головы): каждая голова работает со своей частью чисел
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)       # насколько символы «подходят» друг другу
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        out = self.dropout(weights) @ v                                 # смесь значений с этими весами
        out = out.transpose(1, 2).contiguous().view(B, T, C)            # склеиваем головы обратно
        return self.proj(out)


class Block(nn.Module):
    """Блок трансформера: внимание («посмотреть на соседей») + MLP («подумать»)."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # «+ x» — остаточная связь: блок только добавляет поправку
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Маленькая языковая модель по символам: угадывает следующий символ по block_size предыдущим."""

    def __init__(self, vocab_size, block_size=64, n_embd=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd,
                           n_head=n_head, n_layer=n_layer, dropout=dropout)
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)    # «что за символ»
        self.pos_emb = nn.Embedding(block_size, n_embd)      # «на каком он месте»
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)            # оценки для каждого возможного следующего символа

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T))
        logits = self.head(self.ln(self.blocks(x)))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, generator=None):
        """Дописать n_new символов к началу idx (тензор номеров формы (1, T))."""
        self.eval()
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.block_size:])         # видим не больше block_size символов
            probs = apply_temperature(logits[0, -1], temperature)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
        return idx


def count_params(model):
    """Сколько весов в сети."""
    return sum(p.numel() for p in model.parameters())


def save_gpt(model, tok, name, history=None):
    """Сохранить GPT вместе с настройками и словарём в name.pt, историю — в name.json."""
    torch.save({"config": model.config, "chars": tok.chars, "state": model.state_dict()}, HERE / f"{name}.pt")
    if history:
        (HERE / f"{name}.json").write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
    print(f"Модель сохранена в {name}.pt")


def load_gpt(name):
    """Загрузить GPT и словарь, сохранённые save_gpt. Возвращает (model, tok)."""
    path = HERE / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"Нет файла {path.name} — сначала выполните шаг, который его обучает")
    saved = torch.load(path)
    tok = CharTokenizer("".join(saved["chars"]))
    model = MiniGPT(**saved["config"])
    model.load_state_dict(saved["state"])
    model.eval()
    return model, tok


def load_session04():
    """Модуль common.py из сессии 4 (набор CatsDogs и сеть SimpleCNN) — для бонусного шага."""
    path = HERE.parent / "session04" / "common.py"
    spec = importlib.util.spec_from_file_location("session04_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def show():
    """Показать графики matplotlib (в тихом режиме NO_GUI только сохраняем)."""
    import matplotlib.pyplot as plt
    if not os.environ.get("NO_GUI"):
        plt.show()
    plt.close("all")
