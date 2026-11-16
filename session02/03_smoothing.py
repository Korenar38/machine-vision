"""Шаг 3. Сглаживание: какой фильтр лучше убирает шум?

Добавляем на фото шум «соль и перец» (случайные белые и чёрные точки)
и сравниваем четыре способа сгладить его.
"""

import cv2
import numpy as np

from common import PIPE, grid, label, load, save, show

img = cv2.resize(load(PIPE), (640, 360))

# Шум: 3% пикселей чёрные, 3% белые
rng = np.random.default_rng(seed=0)
noisy = img.copy()
chance = rng.random(img.shape[:2])
noisy[chance < 0.03] = 0
noisy[chance > 0.97] = 255

k = 5  # размер окна фильтра: 5×5 пикселей, всегда нечётный

mean = cv2.blur(noisy, (k, k))                    # среднее по окну
gauss = cv2.GaussianBlur(noisy, (k, k), 0)        # среднее, но центр важнее краёв
median = cv2.medianBlur(noisy, k)                 # медиана: середина отсортированного окна
bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)  # сглаживает, но бережёт границы

# Что внутри cv2.blur: окно 5×5, каждый вес 1/25 — это и есть свёртка (сессия 5)
kernel = np.ones((k, k)) / (k * k)
manual = cv2.filter2D(noisy, -1, kernel)
print("Разница между cv2.blur и своей свёрткой:", int(np.abs(manual.astype(int) - mean).max()))

collage = grid([
    label(noisy, "noise"), label(mean, "blur (mean)"), label(manual, "filter2D: own kernel"),
    label(gauss, "GaussianBlur"), label(median, "medianBlur"), label(bilateral, "bilateralFilter"),
])

save("smoothing.jpg", collage)
show("smoothing", collage)
print("Вопрос: почему медиана справилась с точками лучше среднего?")
