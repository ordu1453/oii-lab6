import cv2
import pytesseract
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# путь к изображению
IMAGE_PATH = "1/1_9.png"

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
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        # Фильтр по размеру
        if 200 < area < 3000:  # Увеличил минимальную площадь для метки
            marks.append((x, y, cw, ch, area))
    
    if not marks:
        print("Метки не найдены, возвращаю оригинальное изображение")
        return original, []
    
    # Сортируем по положению
    marks.sort(key=lambda m: (m[1], m[0]))
    
    # Отображаем все найденные кандидаты
    debug_img = image.copy()
    for i, (x, y, w, h, area) in enumerate(marks):
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(debug_img, f"{i}:{area:.0f}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # cv2.imshow("Detected candidates", debug_img)
    # cv2.waitKey(0)
    
    # Выбираем нужную метку (обычно вторая метка для верхней левой)
    if len(marks) > 1:
        target_idx = 1  # Вторая метка
    else:
        target_idx = 0
    
    x, y, mark_w, mark_h, area = marks[target_idx]
    
    print(f"Выбрана метка #{target_idx}: x={x}, y={y}, w={mark_w}, h={mark_h}, area={area:.1f}")
    
    # ОТЛАДКА: покажем выбранную метку
    selected_img = image.copy()
    # cv2.rectangle(selected_img, (x, y), (x+mark_w, y+mark_h), (0, 0, 255), 3)
    # cv2.imshow("Selected mark", selected_img)
    # cv2.waitKey(0)
    
    # Ключевое исправление: обрезаем справа от метки и снизу от метки
    # Для обрезки СЛЕВА И СВЕРХУ (как в задании):
    # Обрезаем от правого края метки до конца изображения (по горизонтали)
    # и от нижнего края метки до конца изображения (по вертикали)
    
    # Начальные координаты для обрезки (после метки)
    crop_start_x = x + mark_w + offset  # Начинаем справа от метки
    crop_start_y = y + mark_h + offset  # Начинаем снизу от метки
    
    # # Проверяем, чтобы координаты были в пределах изображения
    # crop_start_x = max(0, min(crop_start_x, w - 1))
    # crop_start_y = max(0, min(crop_start_y, h - 1))
    
    print(f"Координаты обрезки: x_start={crop_start_x}, y_start={crop_start_y}, x_end={w}, y_end={h}")
    
    # # Проверяем, что область обрезки имеет разумный размер
    # if crop_start_x >= w - 10 or crop_start_y >= h - 10:
    #     print(f"Слишком маленькая область для обрезки!")
    #     print(f"crop_start_x={crop_start_x}, crop_start_y={crop_start_y}")
    #     print("Возвращаю оригинальное изображение")
    #     return original, [(x, y, mark_w, mark_h)]
    
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