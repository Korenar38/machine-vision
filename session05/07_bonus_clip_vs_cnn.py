"""Бонус. Текст + картинка: CLIP против нашей CNN из сессии 4.

CLIP переводит и картинку, и текст в вектор из 512 чисел так, что картинка кошки
оказывается рядом с фразой «a photo of a cat». Поэтому он отличает кошек от собак
без всякого обучения на CIFAR — достаточно написать названия классов словами.

Нужны пакет clip и веса data/clip/ViT-B-32.pt (см. Read.md), а также сети из сессии 4.
"""

import warnings

import matplotlib.pyplot as plt
import torch
from torchvision.transforms.v2 import functional as TF

from common import DATA, load_session04, show

try:
    import clip
    from clip.simple_tokenizer import SimpleTokenizer
except ImportError:
    raise SystemExit("Нет пакета clip — установите его по инструкции из Read.md")

CLIP_DIR = DATA / "clip"
if not (CLIP_DIR / "ViT-B-32.pt").exists():
    print(f"Нет весов {CLIP_DIR / 'ViT-B-32.pt'} — скачиваем один раз (~340 МБ, нужен интернет)")

warnings.filterwarnings("ignore", category=FutureWarning)   # безвредное предупреждение torch.jit при загрузке CLIP
model, preprocess = clip.load("ViT-B/32", device="cpu", download_root=str(CLIP_DIR))
model.eval()
print("Весов в CLIP:", sum(p.numel() for p in model.parameters()))

s4 = load_session04()
test = s4.CatsDogs(train=False)
print("Проверочных картинок из сессии 4:", len(test))

# --- 1. Как CLIP режет текст на токены (BPE, как в шаге 1)
tokenizer = SimpleTokenizer()
for phrase in ["a photo of a cat", "фотография кошки"]:
    ids = tokenizer.encode(phrase)
    pieces = [tokenizer.decode([i]) for i in ids]
    print(f"«{phrase}» → {len(ids)} токенов: {pieces}")
print("Английский CLIP видел мало русского текста: кириллица режется на мелкие кусочки байтов.")


@torch.no_grad()
def encode_images(images):
    """Картинки (N, 3, 32, 32) → векторы CLIP длиной 1 (N, 512)."""
    features = []
    for i in range(0, len(images), 100):
        batch = torch.stack([preprocess(TF.to_pil_image(image)) for image in images[i:i + 100]])
        f = model.encode_image(batch)          # 32×32 растягивается до 224×224 — CLIP учили на таком размере
        features.append(f / f.norm(dim=-1, keepdim=True))
        print(f"\r  картинок закодировано: {i + len(batch)}/{len(images)}", end="")
    print()
    return torch.cat(features)


@torch.no_grad()
def encode_texts(texts):
    """Фразы → векторы CLIP длиной 1 (len(texts), 512)."""
    f = model.encode_text(clip.tokenize(texts))
    return f / f.norm(dim=-1, keepdim=True)


# Косинусная близость на игрушечных векторах из двух чисел: «пушистость» и «лай»
toy = {"кошка": [0.9, 0.1], "котёнок": [1.0, 0.0], "собака": [0.3, 0.9], "машина": [-0.7, 0.2]}
toy = {word: torch.tensor(v) / torch.tensor(v).norm() for word, v in toy.items()}   # длина 1
for word in ["котёнок", "собака", "машина"]:
    print(f"близость «кошка» и «{word}»: {toy['кошка'] @ toy[word]:+.2f}")
print("1 — смотрят в одну сторону, 0 — не связаны, −1 — противоположны")

print("Кодируем картинки (на процессоре это займёт около минуты)...")
image_features = encode_images(test.images)

# --- 2. Классификация без обучения: какая фраза ближе к картинке
results = {}
for prompts in [["cat", "dog"], ["a photo of a cat", "a photo of a dog"],
                ["кошка", "собака"], ["фотография кошки", "фотография собаки"]]:
    similarity = image_features @ encode_texts(prompts).T      # косинусная близость, (2000, 2)
    acc = (similarity.argmax(1) == test.labels).float().mean().item()
    results[f"CLIP: «{prompts[0]}»"] = acc
    print(f"CLIP с подписями {prompts}: точность {acc:.1%}")

# Как выглядят сами числа: близости отличаются совсем чуть-чуть, поэтому CLIP умножает их на 100 перед softmax
similarity = image_features[:4] @ encode_texts(["a photo of a cat", "a photo of a dog"]).T
for i, (cat, dog) in enumerate(similarity.tolist()):
    p_cat = torch.softmax(100 * torch.tensor([cat, dog]), dim=0)[0]
    print(f"картинка {i} ({s4.CLASSES_RU[test.labels[i]]}): близость к cat {cat:.3f}, к dog {dog:.3f} → P(кошка) = {p_cat:.0%}")

for name, dropout in [("mlp", None), ("cnn", 0.0), ("cnn_aug", 0.3)]:
    if not (s4.HERE / f"{name}.pt").exists():
        print(f"Нет session04/{name}.pt — выполните шаги сессии 4, чтобы сравнить")
        continue
    net = s4.MLP() if dropout is None else s4.SimpleCNN(dropout=dropout)
    acc = s4.accuracy(s4.load_model(net, name), test)
    results[f"сессия 4: {name}"] = acc
    print(f"Наша сеть {name} из сессии 4: точность {acc:.1%}")

plt.figure(figsize=(8, 4))
names = list(results)
plt.barh(names, [results[n] for n in names], color=["tab:orange" if n.startswith("CLIP") else "tab:blue" for n in names])
for i, n in enumerate(names):
    plt.text(results[n] + 0.005, i, f"{results[n]:.1%}", va="center")
plt.axvline(0.5, color="gray", linestyle=":")
plt.xlim(0.4, 1.0)
plt.xlabel("точность на проверочных картинках")
plt.title("Кошки против собак: CLIP без обучения и наши сети")
plt.tight_layout()
plt.savefig("clip_vs_cnn.png", dpi=100)
show()

# --- 3. Больше классов, чем две: CLIP можно спросить о чём угодно
labels = ["a photo of a cat", "a photo of a dog", "a photo of a horse", "a photo of a car", "a photo of a frog"]
text_features = encode_texts(labels)
fig, axes = plt.subplots(2, 4, figsize=(12, 7.2))
for ax, i in zip(axes.flat, range(0, 800, 100)):
    probs = (100 * image_features[i] @ text_features.T).softmax(0)   # 100 — «температура» CLIP
    ax.imshow(test.images[i].permute(1, 2, 0))
    best = probs.argsort(descending=True)[:2].tolist()             # две самые вероятные подписи
    ax.set_title(", ".join(f"{labels[j].split()[-1]} {probs[j]:.0%}" for j in best), fontsize=10)
    ax.axis("off")
fig.suptitle("CLIP выбирает подпись из пяти вариантов")
plt.tight_layout()
plt.savefig("clip_labels.png", dpi=100)
show()

# --- 4. Поиск картинок по тексту
queries = ["a black cat", "a white dog", "a cat lying on a bed", "a dog on the grass", "a dog's face close up"]
similarity = image_features @ encode_texts(queries).T
fig, axes = plt.subplots(len(queries), 5, figsize=(9, 2 * len(queries)))
for row, query in enumerate(queries):
    best = similarity[:, row].argsort(descending=True)[:5]
    for col, i in enumerate(best.tolist()):
        ax = axes[row, col]
        ax.imshow(test.images[i].permute(1, 2, 0))
        ax.axis("off")
        if col == 0:
            ax.set_title(f"«{query}»", loc="left", fontsize=10)
fig.suptitle("Поиск по тексту: 5 самых подходящих картинок")
plt.tight_layout()
plt.savefig("clip_search.png", dpi=100)
show()
