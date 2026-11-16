"""Шаг 5. Нарисуй цифру мышкой — сеть угадывает на лету.

Мышь (левая кнопка) — рисовать, c — стереть, p — включить/выключить
подготовку «как в MNIST», q — выход.
"""

import cv2
import numpy as np
import torch

from common import load_model

model = load_model()

SIZE = 420
canvas = np.zeros((SIZE, SIZE), np.uint8)
drawing = False
prepare = True


def on_mouse(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    if drawing and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
        cv2.circle(canvas, (x, y), 16, 255, -1)


def to_mnist(img):
    """Сделать из рисунка картинку 28×28, похожую на цифры MNIST.

    В MNIST цифра вписана в квадрат 20×20 и стоит центром масс посередине.
    Без этой подготовки сеть ошибается гораздо чаще — проверьте клавишей p.
    """
    if not prepare:
        return cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(img)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)
    digit = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # обрезать пустые поля
    h, w = digit.shape
    k = 20 / max(h, w)
    digit = cv2.resize(digit, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.uint8)
    h, w = digit.shape
    out[(28 - h) // 2:(28 - h) // 2 + h, (28 - w) // 2:(28 - w) // 2 + w] = digit
    m = cv2.moments(out)  # сдвинуть центр масс в точку (14, 14)
    shift = np.float32([[1, 0, 14 - m["m10"] / m["m00"]], [0, 1, 14 - m["m01"] / m["m00"]]])
    return cv2.warpAffine(out, shift, (28, 28))


def predict(small):
    x = torch.tensor(small, dtype=torch.float32).div(255).view(1, 1, 28, 28)
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].numpy()  # 10 вероятностей, в сумме 1


def panel(small, probs):
    """Правая панель: что видит сеть (28×28) и столбики уверенности."""
    p = np.full((SIZE, 300, 3), 30, np.uint8)
    p[12:124, 94:206] = cv2.cvtColor(cv2.resize(small, (112, 112), interpolation=cv2.INTER_NEAREST),
                                     cv2.COLOR_GRAY2BGR)
    cv2.putText(p, "network sees", (96, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    best = int(probs.argmax())
    for d in range(10):
        y = 168 + d * 24
        color = (80, 200, 80) if d == best else (200, 160, 90)
        cv2.putText(p, str(d), (14, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        cv2.rectangle(p, (40, y + 2), (40 + int(probs[d] * 200), y + 18), color, -1)
        cv2.putText(p, f"{probs[d]:.0%}", (246, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.putText(p, f"prepare: {'ON' if prepare else 'OFF'}  [p]", (14, SIZE - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
    return p


cv2.namedWindow("draw a digit")
cv2.setMouseCallback("draw a digit", on_mouse)

while True:
    small = to_mnist(canvas)
    probs = predict(small) if canvas.any() else np.zeros(10)
    left = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(left, "[c] clear  [q] quit", (10, SIZE - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.imshow("draw a digit", np.hstack([left, panel(small, probs)]))

    key = cv2.waitKey(20) & 0xFF
    if key == ord("q") or cv2.getWindowProperty("draw a digit", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord("c"):
        canvas[:] = 0
    if key == ord("p"):
        prepare = not prepare

cv2.destroyAllWindows()
