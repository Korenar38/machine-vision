"""Шаг 1. Готовая модель YOLO находит объекты на фотографии.

Запуск:
    python 01_photo.py                 # пример из интернета (автобус)
    python 01_photo.py мое_фото.jpg    # своё фото
"""

import sys

from ultralytics import YOLO

from labels_ru import ru

# Готовая модель, обученная на 80 классах набора COCO.
# Буква n = nano, самая маленькая и быстрая. Файл скачается сам при первом запуске.
# yolo11n.pt — более новая версия, код тот же.
MODEL = "yolov8n.pt"

source = sys.argv[1] if len(sys.argv) > 1 else "https://ultralytics.com/images/bus.jpg"

model = YOLO(MODEL)
result = model(source, conf=0.25)[0]  # conf — минимальная уверенность, от 0 до 1

print(f"\nНайдено объектов: {len(result.boxes)}")
for box in result.boxes:
    name = result.names[int(box.cls)]
    confidence = float(box.conf)
    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
    print(f"  {ru(name):<16} уверенность {confidence:.0%}   рамка ({x1}, {y1}) – ({x2}, {y2})")

result.save("result.jpg")
print("\nКартинка с рамками сохранена в result.jpg")
