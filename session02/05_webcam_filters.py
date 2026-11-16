"""Шаг 5. Свой «Инстаграм»: фильтры на веб-камере.

Клавиши 1–8 переключают фильтр, s — сохранить кадр, q — выход.
"""

import cv2

from common import label


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def sketch(f):
    """Карандашный рисунок: серое делим на размытый негатив."""
    g = gray(f)
    blurred = cv2.GaussianBlur(255 - g, (21, 21), 0)
    return cv2.divide(g, 255 - blurred, scale=256)


def cartoon(f):
    """Мультфильм: гладкие цвета + чёрные контуры."""
    colors = cv2.bilateralFilter(f, 9, 150, 150)
    edges = cv2.adaptiveThreshold(cv2.medianBlur(gray(f), 7), 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    return cv2.bitwise_and(colors, colors, mask=edges)


FILTERS = {
    ord("1"): ("original", lambda f: f),
    ord("2"): ("grayscale", gray),
    ord("3"): ("negative", lambda f: 255 - f),
    ord("4"): ("smooth", lambda f: cv2.GaussianBlur(f, (21, 21), 0)),
    ord("5"): ("edges", lambda f: cv2.Canny(gray(f), 60, 120)),
    ord("6"): ("mirror", lambda f: cv2.flip(f, 1)),
    ord("7"): ("sketch", sketch),
    ord("8"): ("cartoon", cartoon),
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Камера не открылась")

name, apply = FILTERS[ord("1")]
shot = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    picture = label(apply(frame), f"{name}   [1-8] filter  [s] save  [q] quit")
    cv2.imshow("filters", picture)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key in FILTERS:
        name, apply = FILTERS[key]
    if key == ord("s"):
        shot += 1
        cv2.imwrite(f"filter_{shot:02d}.jpg", picture)

cap.release()
cv2.destroyAllWindows()
