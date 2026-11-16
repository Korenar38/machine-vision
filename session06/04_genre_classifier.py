"""Шаг 4. Определяем жанр текста: своя голова поверх готовой модели.

120 коротких отрывков шести жанров (genres.csv). rubert-tiny2 превращает каждый отрывок в вектор,
а мы учим поверх него один слой Linear — тем же циклом, что в сессии 3. Сравниваем три способа:
подписи жанров без обучения, «средний вектор жанра» и обученный слой. Сохраняет genre_head.pt для шага 8.
"""

import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from common import HERE, Embedder, load_genres, onegin_stanzas, show

torch.manual_seed(0)
random.seed(0)

rows = load_genres()
GENRES = list(dict.fromkeys(r["genre"] for r in rows))      # жанры в порядке появления в файле
train = [r for r in rows if r["split"] == "train"]
test = [r for r in rows if r["split"] == "test"]
print("жанры:", GENRES)
print(f"для обучения: {len(train)} отрывков, для проверки: {len(test)}")
for g in GENRES:
    example = next(r["text"] for r in train if r["genre"] == g)
    print(f"\n[{g}] {example}")

embed = Embedder()
X_train = embed([r["text"] for r in train])
X_test = embed([r["text"] for r in test])
y_train = torch.tensor([GENRES.index(r["genre"]) for r in train])
y_test = torch.tensor([GENRES.index(r["genre"]) for r in test])
print("векторы для обучения:", tuple(X_train.shape), " для проверки:", tuple(X_test.shape))


def accuracy(pred, y):
    return (pred == y).float().mean().item()

descriptions = {
    "сказка": "Это русская народная сказка.",
    "детектив": "Это детектив о расследовании преступления.",
    "фантастика": "Это научная фантастика о космосе и будущем.",
    "стихи": "Это стихотворение.",
    "новости": "Это новость.",
    "научпоп": "Это научно-популярный текст.",
}
D = embed([descriptions[g] for g in GENRES])
pred0 = (X_test @ D.T).argmax(1)
print(f"Способ 0, по подписям: точность {accuracy(pred0, y_test):.0%}  (наугад: {1 / len(GENRES):.0%})")

centroids = torch.stack([X_train[y_train == k].mean(0) for k in range(len(GENRES))])
centroids = F.normalize(centroids, dim=-1)
pred1 = (X_test @ centroids.T).argmax(1)
print(f"Способ 1, средний вектор: точность {accuracy(pred1, y_test):.0%}")

head = nn.Linear(312, len(GENRES))
optimizer = torch.optim.Adam(head.parameters(), lr=0.003, weight_decay=1e-3)
SCALE = 10
history = {"train": [], "test": []}
for epoch in range(200):
    loss = F.cross_entropy(head(SCALE * X_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        history["train"].append(loss.item())
        history["test"].append(F.cross_entropy(head(SCALE * X_test), y_test).item())
    if epoch % 40 == 0 or epoch == 199:
        print(f"эпоха {epoch:3d}: loss на обучении {history['train'][-1]:.3f}, на проверке {history['test'][-1]:.3f}")

with torch.no_grad():
    pred2 = head(SCALE * X_test).argmax(1)
print(f"\nСпособ 2, обученная голова: точность {accuracy(pred2, y_test):.0%}")
print(f"весов в голове: {sum(p.numel() for p in head.parameters())} (в BERT под ней — 29 млн, их мы не трогали)")
torch.save({"genres": GENRES, "scale": SCALE, "state": head.state_dict()}, HERE / "genre_head.pt")
print("голова сохранена в genre_head.pt")

plt.figure(figsize=(7, 4))
plt.plot(history["train"], label="обучение")
plt.plot(history["test"], label="проверка")
plt.xlabel("эпоха")
plt.ylabel("loss")
plt.title("Обучение головы жанров")
plt.legend()
plt.tight_layout()
plt.savefig("genre_training.png", dpi=100)
show()

confusion = torch.zeros(len(GENRES), len(GENRES), dtype=torch.int)
for t, p in zip(y_test, pred2):
    confusion[t, p] += 1

plt.figure(figsize=(6, 5))
plt.imshow(confusion, cmap="Blues")
plt.xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
plt.yticks(range(len(GENRES)), GENRES)
for i in range(len(GENRES)):
    for j in range(len(GENRES)):
        plt.text(j, i, int(confusion[i, j]), ha="center", va="center", color="white" if confusion[i, j] > 2 else "black")
plt.xlabel("ответ модели")
plt.ylabel("настоящий жанр")
plt.title("Матрица ошибок на проверке")
plt.tight_layout()
plt.savefig("genre_confusion.png", dpi=100)
show()

for r, t, p in zip(test, y_test, pred2):
    if t != p:
        print(f"[{GENRES[t]} → {GENRES[p]}] {r['text'][:110]}")

def train_head(X, y, epochs=200):
    h = nn.Linear(X.shape[1], len(GENRES))
    opt = torch.optim.Adam(h.parameters(), lr=0.003, weight_decay=1e-3)
    for _ in range(epochs):
        loss = F.cross_entropy(h(SCALE * X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return h


shots = [1, 2, 3, 5, 10, 15]
curves = {"средний вектор": [], "обученная голова": []}
for k in shots:
    acc_c, acc_h = [], []
    for trial in range(20):
        idx = torch.cat([torch.nonzero(y_train == g).view(-1)[torch.randperm(15)[:k]] for g in range(len(GENRES))])
        Xk, yk = X_train[idx], y_train[idx]
        c = F.normalize(torch.stack([Xk[yk == g].mean(0) for g in range(len(GENRES))]), dim=-1)
        acc_c.append(accuracy((X_test @ c.T).argmax(1), y_test))
        h = train_head(Xk, yk)
        with torch.no_grad():
            acc_h.append(accuracy(h(SCALE * X_test).argmax(1), y_test))
    curves["средний вектор"].append(sum(acc_c) / len(acc_c))
    curves["обученная голова"].append(sum(acc_h) / len(acc_h))
    print(f"{k:2d} примеров на жанр: средний вектор {curves['средний вектор'][-1]:.0%}, голова {curves['обученная голова'][-1]:.0%}")

plt.figure(figsize=(7, 4))
for name, values in curves.items():
    plt.plot(shots, values, marker="o", label=name)
plt.axhline(accuracy(pred0, y_test), color="gray", linestyle="--", label="по подписям (0 примеров)")
plt.axhline(1 / len(GENRES), color="gray", linestyle=":", label="наугад")
plt.xlabel("примеров каждого жанра для обучения")
plt.ylabel("точность на проверке")
plt.ylim(0, 1.05)
plt.title("Сколько примеров нужно, чтобы различать жанры")
plt.legend()
plt.tight_layout()
plt.savefig("genre_shots.png", dpi=100)
show()

def predict(text):
    with torch.no_grad():
        probs = torch.softmax(head(SCALE * embed(text)), dim=-1)[0]
    best = probs.argsort(descending=True)[:2].tolist()
    return ", ".join(f"{GENRES[i]} {probs[i]:.0%}" for i in best)


stanza = onegin_stanzas()[0]
others = {
    "«Онегин», первая строфа": stanza,
    "Пушкин, «Сказка о рыбаке и рыбке»": "Жил старик со своею старухой у самого синего моря; они жили в ветхой землянке ровно тридцать лет и три года.",
    "из учебника биологии": "Митохондрии — органоиды клетки, в которых происходит клеточное дыхание и вырабатывается энергия.",
    "спортивная новость": "Футболисты «Зенита» обыграли «Спартак» со счётом 2:1 и вышли на первое место в турнирной таблице.",
    "школьное сочинение": "Летом я ездил к бабушке в деревню. Мы ходили на речку, собирали малину и смотрели на звёзды.",
    "рецепт": "Возьмите три яйца, стакан муки и стакан сахара. Взбейте яйца с сахаром и аккуратно добавьте муку.",
}
for name, text in others.items():
    print(f"{name:35s} → {predict(text)}")

from common import load_tokenizer

bert_tok = load_tokenizer("bert")
print("с переводом строки:", bert_tok("Мой дядя,\nсамых честных")["input_ids"])
print("с пробелом:        ", bert_tok("Мой дядя, самых честных")["input_ids"], "— одно и то же")

stanzas = onegin_stanzas()[:60]
with torch.no_grad():
    answers = head(SCALE * embed(stanzas)).argmax(1)
counts = {g: (answers == i).sum().item() for i, g in enumerate(GENRES)}
print(f"\nПервые {len(stanzas)} строф «Онегина» по мнению модели:", {g: c for g, c in counts.items() if c})
