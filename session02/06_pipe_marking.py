"""Шаг 6. Связь с финальным проектом: делаем маркировку трубы читаемой.

Цепочка: вырезать → увеличить → повернуть → серый → контраст → сгладить → порог.
"""

import cv2
import numpy as np

from common import PIPE, label, load, save, show

img = load(PIPE)

crop = img[90:240, 560:1260]                                  # 1. вырезать область маркировки
big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # 2. увеличить в 2 раза
rotated = cv2.rotate(big, cv2.ROTATE_180)                     # 3. текст был вверх ногами
gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)              # 4. цвет не нужен
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))   # 5. контраст по кусочкам:
contrast = clahe.apply(gray)                                  #    блики не мешают тёмным местам
smooth = cv2.medianBlur(contrast, 3)                          # 6. убрать мелкие точки ржавчины
binary = cv2.adaptiveThreshold(smooth, 255,                   # 7. свой порог для каждого участка
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5)

steps = np.vstack([
    label(big, "1-2. crop + resize x2"),
    label(rotated, "3. rotate 180"),
    label(gray, "4. grayscale"),
    label(contrast, "5. CLAHE contrast"),
    label(binary, "6-7. median + adaptive threshold"),
])

save("marking_steps.jpg", steps)
show("pipe marking", steps)
print("Вопрос: на каком шаге текст читается лучше всего? Почему не на последнем?")
