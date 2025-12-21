import cv2
import pytesseract
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# путь к изображению
IMAGE_PATH = "1/1_1.png"
PATH = "res.png"

def crop_by_marks(image, offset=50):
    """
    Обрезает изображение, удаляя верхнюю левую метку позиционирования.
    
    Args:
        image: исходное изображение
        offset: дополнительный отступ после метки (положительный = отступ от метки)
        
    Returns:
        cropped_image: обрезанное изображение
        mark_coords: координаты найденной метки (x, y, w, h)
    """
    # Создаем копию изображения
    original = image.copy()
    
    # Конвертируем в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    print(f"Размер изображения: {image.shape}")
    
    # Бинаризуем изображение
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marks = []
    
    # Определяем область поиска для верхней левой метки (например, 1/4 изображения)
    search_area_w = w // 2  # левая половина по ширине
    search_area_h = h // 2  # верхняя половина по высоте
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Фильтр по размеру
        if 200 < area < 3000:
            # Фильтр по положению: только в верхней левой части
            if x < search_area_w and y < search_area_h:
                marks.append((x, y, cw, ch, area))
    
    if not marks:
        print("Метки в верхнем левом углу не найдены, возвращаю оригинальное изображение")
        return original, []
    
    # Сортируем метки по положению (сначала по Y, потом по X)
    marks.sort(key=lambda m: (m[1], m[0]))
    
    # Отображаем все найденные кандидаты
    debug_img = image.copy()
    # Рисуем область поиска
    cv2.rectangle(debug_img, (0, 0), (search_area_w, search_area_h), (255, 0, 0), 2)
    
    for i, (x, y, w, h, area) in enumerate(marks):
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{i}:{area:.0f}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # cv2.imshow("Detected candidates in top-left area", debug_img)
    # cv2.waitKey(0)
    
    # Выбираем самую верхнюю левую метку (первую после сортировки)
    x, y, mark_w, mark_h, area = marks[0]
    
    print(f"Найдено меток в верхнем левом углу: {len(marks)}")
    print(f"Выбрана верхняя левая метка: x={x}, y={y}, w={mark_w}, h={mark_h}, area={area:.1f}")
    
    # ОТЛАДКА: покажем выбранную метку
    selected_img = image.copy()
    cv2.rectangle(selected_img, (x, y), (x+mark_w, y+mark_h), (0, 0, 255), 3)
    cv2.rectangle(selected_img, (0, 0), (search_area_w, search_area_h), (255, 0, 0), 2)
    # cv2.imshow("Selected top-left mark", selected_img)
    # cv2.waitKey(0)
    
    # Вычисляем координаты для обрезки
    # Обрезаем справа от метки и снизу от метки
    crop_start_x = x + mark_w + offset
    crop_start_y = y + mark_h + offset
    
    print(f"Координаты обрезки: x_start={crop_start_x}, y_start={crop_start_y}, x_end={w}, y_end={h}")
    
    
    # Обрезаем изображение (от crop_start_x до конца по ширине, от crop_start_y до конца по высоте)
    roi = image[crop_start_y:, crop_start_x:]
    
    print(f"Размер ROI: {roi.shape if roi.size > 0 else 'пустой'}")
    
    # Проверяем, что ROI не пустой
    if roi.size == 0:
        print("ROI пустой, возвращаю оригинальное изображение")
        return original, [(x, y, mark_w, mark_h)]
    
    # Покажем результат обрезки
    # cv2.imshow("Cropped image", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return roi, [(x, y, mark_w, mark_h)]

# Дополнительная функция для визуализации меток (для отладки)
def draw_marks(image, marks, color=(0, 255, 0), thickness=2):
    """Рисует найденные метки на изображении"""
    debug_img = image.copy()
    for x, y, w, h in marks:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, thickness)
    return debug_img

# -------------------------------
# 1. Загрузка
# -------------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Ошибка: Не удалось загрузить изображение {IMAGE_PATH}")
    exit(1)

print(f"Размер оригинального изображения: {img.shape}")

# -------------------------------
# 2. Автообрезка по меткам
# -------------------------------

roi, marks = crop_by_marks(img)
print(f"Найдено меток: {len(marks)}")
print(f"Размер ROI: {roi.shape}")

draw = draw_marks(img, marks)

cv2.imwrite("marks.png", draw)

# Проверяем, что ROI не пустой перед сохранением
if roi is not None and roi.size > 0:
    cv2.imwrite("roi_without_marks.png", roi)
    print("Обрезанное изображение сохранено как roi_without_marks.png")
    
    # Показать результат (опционально)
    # cv2.imshow("Original", img)
    # cv2.imshow("ROI", roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    print("Ошибка: ROI пустой, пропускаем обрезку")
    roi = img.copy()

# -------------------------------
# 3. Предобработка
# -------------------------------
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

gray = cv2.blur(gray,(5,5))

_, thresh = cv2.threshold(
    gray, 100, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# # Конвертация в grayscale
# gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# # Повышение контраста (CLAHE)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# enhanced = clahe.apply(gray)

# # Бинаризация
# _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# # Удаление шума
# denoised = cv2.medianBlur(thresh, 3)

# Сохраняем бинаризованное изображение для отладки
cv2.imwrite("binary.png", thresh)

# -------------------------------
# 4. OCR
# -------------------------------
config = r"--oem 3 --psm 1 -l rus"
raw_text = pytesseract.image_to_string(thresh, config=config)

lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
print(f"Распознано строк: {len(lines)}")
# print(raw_text)

# Создаем имя текстового файла
text_file = os.path.splitext(PATH)[0] + ".txt"

# Записываем текст в файл
with open(text_file, "w", encoding="utf-8") as file:
    file.write(raw_text)

# -------------------------------
# 5. Эталон
# -------------------------------
expected_lines = []

expected_lines.append(
    "Задание указано на доске. Подписывать бланк НЕ требуется"
)

for _ in range(5):
    expected_lines.append("АБВГДЕЁЖЗ ИЙКЛМНОПРСТ У")
    expected_lines.append("ФХЦЧШЩЪЫЬЭЮЯ-0123456789")

for _ in range(5):
    expected_lines.append("АБВГДЕЁЖЗИЙ КЛМНОПРСТ УФХЦЧШЩЪЫЬЭ")
    expected_lines.append("ЮЯ-0123456789")

# -------------------------------
# 6. Метрики
# -------------------------------
def char_accuracy(gt, pred):
    return SequenceMatcher(None, gt, pred).ratio()

print("\n=== OCR С АВТООБРЕЗКОЙ ПО МЕТКАМ ===\n")

acc_list, cer_list = [], []

# Используем минимальное количество строк для сравнения
min_lines = min(len(expected_lines), len(lines))

if min_lines == 0:
    print("Ошибка: Нет строк для сравнения")
else:
    for i in range(min_lines):
        gt = expected_lines[i]
        pred = lines[i] if i < len(lines) else ""
        acc = char_accuracy(gt, pred)
        err = cer(gt, pred)

        acc_list.append(acc)
        cer_list.append(err)

        print(f"Строка {i}")
        print(f"GT  : {gt}")
        print(f"OCR : {pred}")
        print(f"ACC : {acc:.3f}")
        print(f"CER : {err:.3f}")
        print("-" * 50)

    print("\n=== ИТОГО ===")
    print(f"Средняя Accuracy: {sum(acc_list)/len(acc_list):.4f}")
    print(f"Средний CER: {sum(cer_list)/len(cer_list):.4f}")