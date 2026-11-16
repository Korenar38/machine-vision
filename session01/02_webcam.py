"""Шаг 2. YOLO смотрит в веб-камеру в реальном времени.

Клавиши в окне:
    +  / -   поднять / опустить порог уверенности
    s        сохранить кадр (пригодится для таблицы ошибок)
    q        выход
"""

import time
from collections import Counter

import cv2
from ultralytics import YOLO

from labels_ru import ru

MODEL = "yolov8n.pt"
CAMERA = 0  # если камер несколько, попробуйте 1

model = YOLO(MODEL)
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    raise SystemExit("Камера не открылась: проверьте подключение или номер CAMERA")

conf = 0.40
shot = 0
last_report = ""
prev = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, conf=conf, verbose=False)[0]
    picture = result.plot()  # кадр с нарисованными рамками и подписями

    # Скорость: сколько кадров в секунду успевает обработать компьютер
    now = time.time()
    fps = 1 / (now - prev)
    prev = now
    cv2.putText(picture, f"FPS {fps:4.1f}   conf {conf:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Что в кадре — по-русски в консоль, только когда состав меняется
    counts = Counter(result.names[int(c)] for c in result.boxes.cls)
    report = ", ".join(f"{ru(n)}: {k}" for n, k in counts.most_common())
    if report != last_report:
        print(report or "никого")
        last_report = report

    cv2.imshow("YOLO", picture)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key in (ord("+"), ord("=")):
        conf = min(conf + 0.05, 0.95)
    if key == ord("-"):
        conf = max(conf - 0.05, 0.05)
    if key == ord("s"):
        shot += 1
        cv2.imwrite(f"shot_{shot:02d}.jpg", picture)
        print(f"Сохранено: shot_{shot:02d}.jpg")

cap.release()
cv2.destroyAllWindows()
