"""Шаг 1. Картинка — это массив чисел NumPy.

Любая клавиша закрывает окно и переходит к следующему.
"""

import cv2

from common import PIPE, load, save, show

img = load(PIPE)

print("Тип:", type(img))
print("Форма (высота, ширина, каналы):", img.shape)
print("Тип чисел:", img.dtype, "— каждое число от 0 до 255")
print("Всего чисел в картинке:", img.size)

# Один пиксель — три числа. Внимание: OpenCV хранит цвета в порядке B, G, R
y, x = 760, 1000  # точка на зелёной краске подставки
blue, green, red = img[y, x]
print(f"Пиксель ({x}, {y}): синий={blue}, зелёный={green}, красный={red}")

# Срез массива = вырезать кусок картинки: [строки y, столбцы x]
crop = img[90:240, 560:1260]
print("Кусок с маркировкой:", crop.shape)

# Изменить числа = изменить картинку
marked = img.copy()
marked[90:240, 560:1260, 2] = 255  # канал 2 (красный) в области маркировки на максимум

# Каналы по отдельности: каждый — серая картинка
b, g, r = cv2.split(img)
print("Средняя яркость каналов B, G, R:", int(b.mean()), int(g.mean()), int(r.mean()))

show("crop", crop)
show("marked", marked)
show("red channel", r)
save("crop.jpg", crop)
