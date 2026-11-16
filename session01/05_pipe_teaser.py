"""Финал урока. Покажем готовой модели фото трубы с маркировкой.

Модель знает 80 бытовых классов, маркировки среди них нет —
она её не найдёт. Через несколько сессий мы научим свою модель.
"""

from pathlib import Path

from ultralytics import YOLO

from labels_ru import ru

PIPE = Path(__file__).parent.parent / "dataset" / "images" / "train" / "C20230119_093518708.jpg"

model = YOLO("yolov8n.pt")
result = model(str(PIPE), conf=0.25)[0]

if len(result.boxes) == 0:
    print("Модель ничего не нашла на фото трубы.")
for box in result.boxes:
    print(f"Модель думает, что это: {ru(result.names[int(box.cls)])} ({float(box.conf):.0%})")

result.save("pipe_result.jpg")
print("Вопрос классу: почему она не видит маркировку? Что ей нужно, чтобы научиться?")
