import cv2
import pytesseract
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# путь к изображению
IMAGE_PATH = "1/1_8.png"

def crop_by_marks(image, offset=100):
    """
    Обрезает изображение, удаляя верхнюю левую метку позиционирования.
    Метка - это Х в кружке.
    """
    # Создаем копию изображения
    original = image.copy()
    
    # Конвертируем в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Бинаризуем изображение
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marks = []
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Фильтр по размеру
        if 50 < area < 3000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                # Фильтруем по кругости
                if circularity > 0.4:
                    marks.append((x, y, cw, ch, area, circularity))
    
    if not marks:
        print("Метки не найдены, возвращаю оригинальное изображение")
        return original, []
    
    # Сортируем по положению (верхний левый угол)
    marks.sort(key=lambda m: (m[1], m[0]))
    
    # Отображаем все найденные кандидаты
    debug_img = image.copy()
    for i, (x, y, w, h, area, circ) in enumerate(marks):
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{i}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("Detected candidates", debug_img)
    cv2.waitKey(0)
    
    # Выбираем метку (в вашем случае - вторая метка, индекс 1)
    if len(marks) > 1:
        # Если есть несколько меток, берем вторую (индекс 1)
        target_mark = marks[1]  # Вторая метка в списке
        print(f"Выбрана вторая метка (индекс 1)")
    else:
        target_mark = marks[0]  # Если только одна
        print(f"Выбрана единственная метка")
    
    x, y, mark_w, mark_h, area, circularity = target_mark
    
    print(f"Выбрана метка: x={x}, y={y}, w={mark_w}, h={mark_h}, "
          f"area={area:.1f}, circ={circularity:.2f}")
    
    # Определяем область для обрезки
    crop_x = min(x + mark_w + offset, w - 1)
    crop_y = min(y + mark_h + offset, h - 1)
    
    # Проверяем корректность координат
    if crop_x >= w or crop_y >= h or crop_x < 0 or crop_y < 0:
        print(f"Некорректные координаты обрезки: x={crop_x}, y={crop_y}")
        return original, [(x, y, mark_w, mark_h)]
    
    # Обрезаем изображение
    roi = image[crop_y:h, crop_x:w]
    
    print(f"Обрезано: от y={crop_y} до {h}, от x={crop_x} до {w}")
    
    # Показываем результат обрезки
    cv2.imshow("Cropped result", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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

_, thresh = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Сохраняем бинаризованное изображение для отладки
cv2.imwrite("binary.png", thresh)

# -------------------------------
# 4. OCR
# -------------------------------
config = r"--oem 3 --psm 6 -l rus"
raw_text = pytesseract.image_to_string(thresh, config=config)

lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
print(f"Распознано строк: {len(lines)}")

# -------------------------------
# 5. Эталон
# -------------------------------
expected_lines = []

expected_lines.append(
    "Задание указано на доске. Подписывать бланк не требуется"
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