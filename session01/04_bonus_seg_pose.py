"""Бонус. Та же камера, но другие задачи: контуры объектов и скелет человека.

    python 04_bonus_seg_pose.py seg    # сегментация: точный контур, а не рамка
    python 04_bonus_seg_pose.py pose   # поза: 17 точек тела
"""

import sys

import cv2
from ultralytics import YOLO

task = sys.argv[1] if len(sys.argv) > 1 else "pose"
model = YOLO("yolov8n-seg.pt" if task == "seg" else "yolov8n-pose.pt")

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    picture = model(frame, verbose=False)[0].plot()
    cv2.imshow(f"YOLO {task}", picture)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
