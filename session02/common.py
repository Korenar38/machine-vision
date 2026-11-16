"""Общие помощники для сессии 2: чтение, показ и склейка картинок."""

import os
from pathlib import Path

import cv2
import numpy as np

DATASET = Path(__file__).parent.parent / "dataset" / "images" / "train"
PIPE = DATASET / "C20230119_093518708.jpg"


def load(path):
    """Прочитать картинку.

    cv2.imread на Windows не понимает русские буквы в пути,
    поэтому читаем байты через NumPy и декодируем их сами.
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Не удалось открыть {path}")
    return img


def save(path, img):
    """Сохранить картинку (тоже работает с русскими буквами в пути)."""
    ok, buf = cv2.imencode(Path(path).suffix, img)
    buf.tofile(str(path))
    print(f"Сохранено: {path}")


def to_bgr(img):
    """Серую картинку (один канал) превратить в трёхканальную, чтобы склеивать с цветными."""
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def fit(img, max_w=1280, max_h=720):
    """Уменьшить картинку, если она не влезает в экран."""
    h, w = img.shape[:2]
    k = min(max_w / w, max_h / h, 1)
    if k == 1:
        return img
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)


def show(title, img):
    """Показать окно и ждать любую клавишу. Название окна пишите латиницей."""
    if os.environ.get("NO_GUI"):
        return
    cv2.imshow(title, fit(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_nb(img, max_w=1280, max_h=720):
    """Показать картинку прямо в ноутбуке Jupyter (под ячейкой, без отдельного окна)."""
    from IPython.display import Image, display

    ok, buf = cv2.imencode(".png", fit(img, max_w, max_h))
    display(Image(data=buf.tobytes()))


def label(img, text):
    """Подписать картинку сверху (OpenCV рисует только латиницу)."""
    out = to_bgr(img).copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def grid(images, cols=3):
    """Склеить картинки одинакового размера в таблицу."""
    images = [to_bgr(i) for i in images]
    while len(images) % cols:
        images.append(np.zeros_like(images[0]))
    rows = [np.hstack(images[i:i + cols]) for i in range(0, len(images), cols)]
    return np.vstack(rows)
