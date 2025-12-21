import cv2
import pytesseract
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


# путь к изображению
IMAGE_PATH = "1/1_8.png"

def find_alignment_marks(binary_img):
    """
    Находит координаты меток позиционирования
    Возвращает список bounding box'ов (x, y, w, h)
    """
    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = binary_img.shape
    marks = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # фильтр по размеру (подходит под метки)
        if 20 < area < 1000:
            # близко к краям
            if x < w * 0.1 or y < h * 0.1:
                marks.append((x, y, cw, ch))

    return marks

def crop_by_marks(image, offset=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Инвертируем для поиска контуров меток
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    top_cut = 0
    left_cut = 0
    right_cut = w

    marks = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # фильтр по размеру (подходит под метки)
        if 20 < area < 1000:
            marks.append((x, y, cw, ch))

    if marks:
        # верхняя граница ROI
        top_cut = max(y + ch for x, y, cw, ch in marks if y < h * 0.5)

        # левая граница ROI
        left_cut = max(x + cw for x, y, cw, ch in marks if x < w * 0.5)

        # правая граница ROI
        right_cut = min(x for x, y, cw, ch in marks if x > w * 0.5)

    # добавляем небольшой отступ
    top_cut += offset
    left_cut += offset
    right_cut -= offset

    roi = image[top_cut:h, left_cut:right_cut]
    return roi, marks


# -------------------------------
# 1. Загрузка
# -------------------------------
img = cv2.imread(IMAGE_PATH)

# -------------------------------
# 2. Автообрезка по меткам
# -------------------------------
roi, marks = crop_by_marks(img)

cv2.imwrite("roi_without_marks.png", roi)
print("Обрезанное изображение сохранено как roi_without_marks.png")
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
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

lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

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
print(f"Средняя Accuracy: {sum(acc_list)/len(acc_list):.4f}")
print(f"Средний CER: {sum(cer_list)/len(cer_list):.4f}")
