import cv2
import pytesseract
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# путь к изображению
IMAGE_PATH = "1/1_1.png"

# эталонный текст
GROUND_TRUTH = (
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУ\n"
    "ФХЦЧШЩЪЫЬЭЮЯ-0123456789"
)

# -------------------------------
# 1. Предобработка изображения
# -------------------------------
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# бинаризация (лучше для печатных шаблонов)
_, thresh = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# -------------------------------
# 2. OCR
# -------------------------------
custom_config = r"--oem 3 --psm 4 -l rus"

recognized_text = pytesseract.image_to_string(
    thresh,
    config=custom_config
)

# очистка
recognized_text = recognized_text.strip()

print("=== Распознанный текст ===")
print(recognized_text)

# -------------------------------
# 3. Метрики точности
# -------------------------------

# Character Accuracy
def char_accuracy(gt, pred):
    matcher = SequenceMatcher(None, gt, pred)
    return matcher.ratio()

char_acc = char_accuracy(GROUND_TRUTH, recognized_text)
char_error_rate = cer(GROUND_TRUTH, recognized_text)
word_error_rate = wer(GROUND_TRUTH, recognized_text)

print("\n=== Метрики качества ===")
print(f"Character Accuracy: {char_acc:.4f}")
print(f"Character Error Rate (CER): {char_error_rate:.4f}")
print(f"Word Error Rate (WER): {word_error_rate:.4f}")
