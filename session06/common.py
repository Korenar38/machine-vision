"""Общие части сессии 6: загрузка готовых моделей, векторы фраз, набор текстов по жанрам."""

import csv
import os
import warnings
from pathlib import Path

import torch
from torch.nn import functional as F

DATA = Path(__file__).parent.parent / "data"
HERE = Path(__file__).parent
HF_DIR = DATA / "hf"          # сюда скачиваются модели, дальше они берутся без интернета

# Короткие имена моделей сессии → их адрес на Hugging Face Hub
MODELS = {
    "bert": "cointegrated/rubert-tiny2",                              # 29 млн весов, векторы смысла
    "sentiment": "seara/rubert-tiny2-russian-sentiment",              # тон текста
    "emotion": "cointegrated/rubert-tiny2-cedr-emotion-detection",    # эмоции
    "gpt": "ai-forever/rugpt3small_based_on_gpt2",                    # русский GPT, 125 млн весов
    "chat": "Qwen/Qwen2.5-0.5B-Instruct",                             # маленький чат-бот, 500 млн весов
    "chat_big": "Qwen/Qwen2.5-1.5B-Instruct",                         # по желанию (шаг 7): в 3 раза больше, ~3 ГБ
}


def is_downloaded(name):
    """Есть ли модель уже в data/hf (без скачивания)."""
    repo = MODELS.get(name, name)
    return (HF_DIR / repo.replace("/", "__") / "config.json").exists()

warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message="IProgress not found")   # tqdm в Jupyter без ipywidgets — не мешает
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
torch.set_num_threads(min(8, os.cpu_count() or 1))


def model_dir(name):
    """Папка модели в data/hf. Если её нет — скачиваем один раз с Hugging Face Hub (нужен интернет)."""
    repo = MODELS.get(name, name)
    path = HF_DIR / repo.replace("/", "__")
    if not (path / "config.json").exists():
        from huggingface_hub import list_repo_files, snapshot_download
        print(f"Модели {repo} нет в {path} — скачиваем один раз (нужен интернет)...")
        files = list_repo_files(repo)
        skip = ["*.h5", "*.msgpack", "*.ot", "onnx/*", "*.onnx"]
        if any(f.endswith(".safetensors") for f in files):
            skip.append("*.bin")            # одни и те же веса часто лежат в двух форматах — берём один
        snapshot_download(repo, local_dir=path, ignore_patterns=skip)
    return path


def quiet():
    """Убрать служебные сообщения transformers (отчёты о загрузке весов, полоски прогресса)."""
    import transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()


def load_tokenizer(name):
    """Токенизатор (словарь) модели."""
    from transformers import AutoTokenizer
    quiet()
    return AutoTokenizer.from_pretrained(model_dir(name))


def load_model(name, kind="base"):
    """Готовая модель в режиме eval.

    kind: "base" — только тело (выдаёт векторы), "classifier" — с головой-классификатором,
    "lm" — генератор текста (GPT), "mlm" — угадывание пропущенного слова (BERT).
    """
    import transformers
    quiet()
    auto = {"base": transformers.AutoModel,
            "classifier": transformers.AutoModelForSequenceClassification,
            "lm": transformers.AutoModelForCausalLM,
            "mlm": transformers.AutoModelForMaskedLM}[kind]
    model = auto.from_pretrained(model_dir(name), dtype=torch.float32)
    model.eval()
    return model


def count_params(model):
    """Сколько весов в сети."""
    return sum(p.numel() for p in model.parameters())


class Embedder:
    """Фраза → вектор смысла длиной 1 (rubert-tiny2: 312 чисел).

    Вектор фразы — это выход BERT на служебном токене [CLS], нормированный до длины 1,
    так что скалярное произведение двух векторов — косинусная близость (как у CLIP в сессии 5).
    """

    def __init__(self, name="bert"):
        self.tok = load_tokenizer(name)
        self.model = load_model(name)

    @torch.no_grad()
    def __call__(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=256, return_tensors="pt")
            out = self.model(**batch).last_hidden_state[:, 0]     # вектор токена [CLS]
            vectors.append(F.normalize(out, dim=-1))
        return torch.cat(vectors)


def load_genres():
    """Набор отрывков по жанрам из genres.csv: список словарей {"genre", "split", "text"}."""
    with open(HERE / "genres.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_onegin():
    """«Евгений Онегин» из сессии 5 (session05/onegin.txt)."""
    return (HERE.parent / "session05" / "onegin.txt").read_text(encoding="utf-8")


def onegin_stanzas(text=None):
    """Строфы «Онегина» — куски текста между пустыми строками, не короче 4 строк."""
    text = text or load_onegin()
    return [s.strip() for s in text.split("\n\n") if s.strip().count("\n") >= 3]


quiet()


def show():
    """Показать графики matplotlib (в тихом режиме NO_GUI только сохраняем)."""
    import matplotlib.pyplot as plt
    if not os.environ.get("NO_GUI"):
        plt.show()
    plt.close("all")
