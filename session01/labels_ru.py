"""Русские названия классов COCO, на которых обучена готовая модель YOLO.

На картинке OpenCV рисует только латиницу, поэтому по-русски
названия выводятся в консоль.
"""

LABELS_RU = {
    "person": "человек", "bicycle": "велосипед", "car": "машина",
    "motorcycle": "мотоцикл", "bus": "автобус", "truck": "грузовик",
    "traffic light": "светофор", "stop sign": "знак стоп",
    "bench": "скамейка", "bird": "птица", "cat": "кошка",
    "dog": "собака", "backpack": "рюкзак", "umbrella": "зонт",
    "handbag": "сумка", "tie": "галстук", "suitcase": "чемодан",
    "sports ball": "мяч", "baseball bat": "бита", "bottle": "бутылка",
    "cup": "кружка", "fork": "вилка", "knife": "нож", "spoon": "ложка",
    "banana": "банан", "apple": "яблоко", "orange": "апельсин",
    "chair": "стул", "couch": "диван", "potted plant": "цветок в горшке",
    "bed": "кровать", "dining table": "стол", "tv": "телевизор",
    "laptop": "ноутбук", "mouse": "мышь", "remote": "пульт",
    "keyboard": "клавиатура", "cell phone": "телефон", "book": "книга",
    "clock": "часы", "scissors": "ножницы", "teddy bear": "плюшевый мишка",
    "toothbrush": "зубная щётка",
}


def ru(name: str) -> str:
    """Вернуть русское название класса, если оно есть в словаре."""
    return LABELS_RU.get(name, name)
