import cv2
import pytesseract
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# путь к изображению
IMAGE_PATH = "1/1_1.png"

def crop_roi(image, x_min, y_min, x_max, y_max):
    """
    x_min, y_min, x_max, y_max — в долях от 0 до 1
    """
    h, w = image.shape[:2]

    x1 = int(w * x_min)
    y1 = int(h * y_min)
    x2 = int(w * x_max)
    y2 = int(h * y_max)

    return image[y1:y2, x1:x2]

# -------------------------------
# 1. Загрузка изображения
# -------------------------------
img = cv2.imread(IMAGE_PATH)

# -------------------------------
# 2. Ограничение области (ROI)
# -------------------------------
# Подбирается под данный шаблон
roi = crop_roi(
    img,
    x_min=0.11,   # слева
    y_min=0.1,   # сверху
    x_max=0.95,   # справа
    y_max=0.90    # снизу
)

# -------------------------------
# 3. Предобработка
# -------------------------------
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# -------------------------------
# 4. OCR
# -------------------------------
config = r"--oem 3 --psm 6 -l rus"
raw_text = pytesseract.image_to_string(thresh, config=config)

lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

# -------------------------------
# 5. Эталонные строки
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

print("\n=== ПОСТРОЧНАЯ ПРОВЕРКА (ROI) ===\n")

acc_list = []
cer_list = []

for i, (gt, pred) in enumerate(zip(expected_lines, lines)):
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
print(f"Средняя Accuracy: {sum(acc_list) / len(acc_list):.4f}")
print(f"Средний CER: {sum(cer_list) / len(cer_list):.4f}")
