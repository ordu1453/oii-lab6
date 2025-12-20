import cv2
import pytesseract
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# путь к изображению
IMAGE_PATH = "1/1_1.png"

# -------------------------------
# 1. OCR
# -------------------------------
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

config = r"--oem 3 --psm 6 -l rus"
raw_text = pytesseract.image_to_string(thresh, config=config)

# очистка пустых строк
lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

# -------------------------------
# 2. Формирование эталонных строк
# -------------------------------
expected_lines = []

# 0-я строка
expected_lines.append("Задание указано на доске. Подписывать бланк НЕ требуется")

# Первый шаблон (5 повторений)
for _ in range(5):
    expected_lines.append("АБВГДЕЁЖЗ ИЙКЛМНОПРСТ У")
    expected_lines.append("ФХЦЧШЩЪЫЬЭЮЯ-0123456789")

# Второй шаблон (5 повторений)
for _ in range(5):
    expected_lines.append("АБВГДЕЁЖЗИЙ КЛМНОПРСТ УФХЦЧШЩЪЫЬЭ")
    expected_lines.append("ЮЯ-0123456789")

# -------------------------------
# 3. Метрики
# -------------------------------
def char_accuracy(gt, pred):
    return SequenceMatcher(None, gt, pred).ratio()

print("\n=== ПОСТРОЧНАЯ ПРОВЕРКА ===\n")

total_cer = []
total_acc = []

for i, (gt, pred) in enumerate(zip(expected_lines, lines)):
    acc = char_accuracy(gt, pred)
    error = cer(gt, pred)

    total_acc.append(acc)
    total_cer.append(error)

    print(f"Строка {i}:")
    print(f"GT   : {gt}")
    print(f"OCR  : {pred}")
    print(f"ACC  : {acc:.3f}")
    print(f"CER  : {error:.3f}")
    print("-" * 50)

# -------------------------------
# 4. Итоговые метрики
# -------------------------------
print("\n=== ИТОГО ===")
print(f"Средняя Character Accuracy: {sum(total_acc) / len(total_acc):.4f}")
print(f"Средний CER: {sum(total_cer) / len(total_cer):.4f}")

