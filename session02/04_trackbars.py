"""Шаг 4. Пульт управления: ползунки меняют картинку в реальном времени.

Режимы (ползунок mode): 0 — цвет, 1 — серый, 2 — негатив, 3 — порог, 4 — контуры.
Выход: q или Esc.
"""

import cv2

from common import PIPE, fit, load

img = fit(load(PIPE), 960, 540)

WINDOW = "controls"
cv2.namedWindow(WINDOW)


def nothing(_):
    pass


cv2.createTrackbar("brightness", WINDOW, 100, 200, nothing)   # 100 = без изменений
cv2.createTrackbar("contrast x10", WINDOW, 10, 30, nothing)   # 10 = x1.0
cv2.createTrackbar("blur", WINDOW, 0, 15, nothing)
cv2.createTrackbar("mode", WINDOW, 0, 4, nothing)
cv2.createTrackbar("threshold", WINDOW, 120, 255, nothing)

while True:
    beta = cv2.getTrackbarPos("brightness", WINDOW) - 100
    alpha = cv2.getTrackbarPos("contrast x10", WINDOW) / 10
    k = cv2.getTrackbarPos("blur", WINDOW)
    mode = cv2.getTrackbarPos("mode", WINDOW)
    t = cv2.getTrackbarPos("threshold", WINDOW)

    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if k > 0:
        out = cv2.GaussianBlur(out, (2 * k + 1, 2 * k + 1), 0)  # размер окна должен быть нечётным

    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if mode == 1:
        out = gray
    elif mode == 2:
        out = 255 - out
    elif mode == 3:
        _, out = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    elif mode == 4:
        out = cv2.Canny(gray, t // 2, t)

    cv2.imshow(WINDOW, out)
    key = cv2.waitKey(30) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
