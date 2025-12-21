import cv2
import numpy as np

# Загружаем изображение
image = cv2.imread('1/1_1.png')

if image is None:
    print("Ошибка загрузки изображения")
else:
    height, width = image.shape[:2]
    print(f"Исходный размер: {width}x{height}")
    
    # Вариант 1: Удалить 150 пикселей слева
    cropped_1 = image[:, 150:]
    print(f"После обрезки 150px слева: {cropped_1.shape[1]}x{cropped_1.shape[0]}")
    
    # Вариант 2: Оставить только правую половину
    cropped_2 = image[:, width//2:]
    print(f"Только правая половина: {cropped_2.shape[1]}x{cropped_2.shape[0]}")
    
    # Вариант 3: Удалить 25% слева
    cut_px = int(width * 0.25)
    cropped_3 = image[:, cut_px:]
    print(f"Удалено 25% слева: {cropped_3.shape[1]}x{cropped_3.shape[0]}")
    
    # Показать результаты
    cv2.imshow('Original', image)
    cv2.imshow('Cropped 150px left', cropped_1)
    cv2.imshow('Right half', cropped_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()