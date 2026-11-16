"""Шаг 3. Задания: меняем поведение «умной камеры».

Выберите режим в MODE и допишите строки с пометкой ЗАДАНИЕ.
    "count"  — считать людей в кадре
    "filter" — показывать только выбранные классы
    "alarm"  — тревога, если в кадре телефон
"""

import cv2
from ultralytics import YOLO

MODEL = "yolov8n.pt"
MODE = "count"

model = YOLO(MODEL)
cap = cv2.VideoCapture(0)

# Номера классов COCO: 0 — person, 39 — bottle, 41 — cup, 67 — cell phone
# Полный список: print(model.names)
FILTER_CLASSES = [39, 41]  # ЗАДАНИЕ (filter): впишите свои классы

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if MODE == "filter":
        result = model(frame, conf=0.4, classes=FILTER_CLASSES, verbose=False)[0]
    else:
        result = model(frame, conf=0.4, verbose=False)[0]
    picture = result.plot()
    names = [result.names[int(c)] for c in result.boxes.cls]

    if MODE == "count":
        people = names.count("person")
        cv2.putText(picture, f"People: {people}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        # ЗАДАНИЕ (count): если людей больше 3, пишите "Too many!" красным

    if MODE == "alarm":
        if "cell phone" in names:
            h, w = picture.shape[:2]
            cv2.rectangle(picture, (0, 0), (w - 1, h - 1), (0, 0, 255), 20)
            cv2.putText(picture, "PHONE!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        # ЗАДАНИЕ (alarm): срабатывайте, только если телефон виден 30 кадров подряд

    cv2.imshow("YOLO challenge", picture)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
