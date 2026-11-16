"""Шаг 2. Девять базовых операций OpenCV на одной картинке."""

import cv2

from common import PIPE, grid, label, load, save, show

img = cv2.resize(load(PIPE), (640, 360))  # уменьшаем, чтобы коллаж влез в экран

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                  # оттенки серого
negative = 255 - img                                          # негатив, то же что cv2.bitwise_not(img)
smooth = cv2.GaussianBlur(img, (15, 15), 0)                   # сглаживание (размытие)
bright = cv2.convertScaleAbs(img, alpha=1.0, beta=60)         # ярче: +60 к каждому числу
contrast = cv2.convertScaleAbs(img, alpha=1.8, beta=-80)      # контрастнее: умножить на 1.8
rotated = cv2.rotate(img, cv2.ROTATE_180)                     # поворот на 180°
_, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)  # порог: только чёрное и белое
edges = cv2.Canny(gray, 80, 160)                              # границы (контуры)

collage = grid([
    label(img, "original"), label(gray, "grayscale"), label(negative, "negative"),
    label(smooth, "smooth"), label(bright, "brightness +60"), label(contrast, "contrast x1.8"),
    label(rotated, "rotate 180"), label(binary, "threshold 120"), label(edges, "Canny edges"),
])

save("collage.jpg", collage)
show("9 operations", collage)
