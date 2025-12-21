import cv2
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher
import os
import glob
import pandas as pd
from paddleocr import PaddleOCR
import warnings
warnings.filterwarnings("ignore")

# Инициализация PaddleOCR
ocr = PaddleOCR(
    lang='ru'
)

# Папка с изображениями
FOLDER_PATH = "1"

def crop_by_marks(image, offset=50):
    """
    Обрезает изображение, удаляя верхнюю левую метку позиционирования.
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marks = []
    search_area_w = w // 2
    search_area_h = h // 2
    
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        if 200 < area < 3000 and x < search_area_w and y < search_area_h:
            marks.append((x, y, cw, ch, area))
    
    if not marks:
        print("Метки не найдены, возвращаю оригинальное изображение")
        return original, []
    
    marks.sort(key=lambda m: (m[1], m[0]))
    x, y, mark_w, mark_h, area = marks[0]
    
    crop_start_x = x + mark_w + offset
    crop_start_y = y + mark_h + offset
    
    roi = image[crop_start_y:, crop_start_x:]
    
    if roi.size == 0:
        return original, [(x, y, mark_w, mark_h)]
    
    return roi, [(x, y, mark_w, mark_h)]

def draw_marks(image, marks, color=(0, 255, 0), thickness=2):
    debug_img = image.copy()
    for x, y, w, h in marks:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, thickness)
    return debug_img

def char_accuracy(gt, pred):
    return SequenceMatcher(None, gt, pred).ratio()

def preprocess_image_for_ocr(image):
    """
    Предобработка изображения для улучшения качества OCR.
    """
    # Конвертируем в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Несколько методов предобработки
    processed_images = []
    
    # 1. Просто grayscale
    processed_images.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    
    # 2. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    processed_images.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
    
    # 3. Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))
    
    # 4. Улучшение контраста с CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    processed_images.append(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR))
    
    # 5. Медианный фильтр для удаления шума
    median = cv2.medianBlur(gray, 3)
    processed_images.append(cv2.cvtColor(median, cv2.COLOR_GRAY2BGR))
    
    # 6. Гауссово размытие + порог
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, blur_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(cv2.cvtColor(blur_thresh, cv2.COLOR_GRAY2BGR))
    
    return processed_images

def extract_text_with_paddleocr(image, image_name=""):
    """
    Распознает текст с изображения используя PaddleOCR с разными методами предобработки.
    """
    best_text = ""
    best_score = 0
    best_method = "original"
    
    methods = [
        ("original", image),
        ("gray", cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
        ("adaptive", cv2.cvtColor(cv2.adaptiveThreshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2), cv2.COLOR_GRAY2BGR)),
        ("otsu", cv2.cvtColor(cv2.threshold(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], cv2.COLOR_GRAY2BGR)),
        ("denoised", cv2.cvtColor(cv2.fastNlMeansDenoising(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)),
    ]
    
    for method_name, processed_img in methods:
        try:
            print(f"  Пробуем метод: {method_name}")
            
            # Вызываем OCR
            result = ocr.ocr(processed_img)
            
            if result is None or not result:
                print(f"    Метод {method_name}: нет результатов")
                continue
            
            # Извлекаем текст
            text_lines = []
            for page in result:
                if page:
                    for line in page:
                        if line and len(line) >= 2:
                            text = line[1][0]
                            if text and text.strip():
                                text_lines.append(text.strip())
            
            extracted_text = " ".join(text_lines)
            
            # Оцениваем качество (количество символов, наличие русских букв)
            char_count = len(extracted_text)
            russian_chars = sum(1 for c in extracted_text if 'а' <= c <= 'я' or 'А' <= c <= 'Я')
            score = char_count + russian_chars * 5
            
            print(f"    Метод {method_name}: {char_count} символов, {russian_chars} русских букв, score={score}")
            
            if score > best_score:
                best_score = score
                best_text = extracted_text
                best_method = method_name
                
        except Exception as e:
            print(f"    Ошибка в методе {method_name}: {e}")
            continue
    
    print(f"  Выбран метод: {best_method} (score={best_score})")
    return best_text, best_method

def process_image(image_path, save_intermediate=False):
    print(f"\n{'='*60}")
    print(f"Обработка файла: {os.path.basename(image_path)}")
    print('='*60)
    
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return None
    
    # Пробуем несколько способов обработки
    
    # Способ 1: Обрезка по меткам
    roi1, marks1 = crop_by_marks(img)
    
    # Способ 2: Без обрезки (оригинальное изображение)
    roi2 = img.copy()
    
    # Способ 3: Обрезка с другим offset
    roi3, marks3 = crop_by_marks(img, offset=100)
    
    # Сохраняем промежуточные изображения для отладки
    if save_intermediate:
        os.makedirs("debug_paddle", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        cv2.imwrite(f"debug_paddle/{base_name}_original.png", img)
        if marks1:
            draw1 = draw_marks(img, marks1)
            cv2.imwrite(f"debug_paddle/{base_name}_marks.png", draw1)
        cv2.imwrite(f"debug_paddle/{base_name}_roi_offset50.png", roi1)
        cv2.imwrite(f"debug_paddle/{base_name}_roi_offset100.png", roi3)
    
    # Пробуем распознать текст с разных ROI
    print("\nПробуем разные ROI и методы предобработки:")
    
    all_texts = []
    
    # ROI 1 (offset=50)
    print("\n1. ROI с offset=50:")
    text1, method1 = extract_text_with_paddleocr(roi1, f"{os.path.basename(image_path)}_roi1")
    all_texts.append(("roi_offset50", text1))
    
    # ROI 2 (оригинал)
    print("\n2. Оригинальное изображение:")
    text2, method2 = extract_text_with_paddleocr(roi2, f"{os.path.basename(image_path)}_roi2")
    all_texts.append(("original", text2))
    
    # ROI 3 (offset=100)
    print("\n3. ROI с offset=100:")
    text3, method3 = extract_text_with_paddleocr(roi3, f"{os.path.basename(image_path)}_roi3")
    all_texts.append(("roi_offset100", text3))
    
    # Выбираем лучший результат
    best_text = ""
    best_score = 0
    best_method = ""
    
    for roi_type, text in all_texts:
        char_count = len(text)
        russian_chars = sum(1 for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я' or c in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        score = char_count + russian_chars * 10
        
        if score > best_score:
            best_score = score
            best_text = text
            best_method = roi_type
    
    print(f"\nВыбран результат: {best_method} (score={best_score})")
    print(f"Распознанный текст: {best_text}")
    
    # Если текст пустой, пробуем альтернативный подход
    if not best_text.strip():
        print("\nТекст не распознан, пробуем альтернативный подход...")
        
        # Пробуем инвертировать цвета
        inverted = cv2.bitwise_not(img)
        text_inv, _ = extract_text_with_paddleocr(inverted, "inverted")
        all_texts.append(("inverted", text_inv))
        
        # Пробуем увеличить контраст
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        text_enh, _ = extract_text_with_paddleocr(enhanced, "enhanced")
        all_texts.append(("enhanced", text_enh))
        
        # Выбираем лучший из всех
        for roi_type, text in all_texts:
            char_count = len(text)
            russian_chars = sum(1 for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я' or c in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
            score = char_count + russian_chars * 10
            
            if score > best_score:
                best_score = score
                best_text = text
                best_method = roi_type
    
    # Разделяем текст на строки
    lines = []
    if best_text:
        # Пробуем разные способы разделения
        temp_lines = best_text.split('\n')
        if len(temp_lines) > 1:
            lines = [l.strip() for l in temp_lines if l.strip()]
        else:
            # Разделяем по точкам или длинным строкам
            sentences = best_text.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    # Если предложение длинное, разбиваем на части
                    if len(sentence) > 30:
                        words = sentence.split()
                        current_line = []
                        current_len = 0
                        for word in words:
                            if current_len + len(word) > 30 and current_line:
                                lines.append(' '.join(current_line))
                                current_line = [word]
                                current_len = len(word)
                            else:
                                current_line.append(word)
                                current_len += len(word) + 1
                        if current_line:
                            lines.append(' '.join(current_line))
                    else:
                        lines.append(sentence)
    
    print(f"\nРазделено на {len(lines)} строк:")
    for i, line in enumerate(lines):
        print(f"  Строка {i}: {line}")
    
    # Сохраняем результаты
    text_file = os.path.splitext(image_path)[0] + "_paddle.txt"
    with open(text_file, "w", encoding="utf-8") as file:
        file.write(f"Метод: {best_method}\n")
        file.write(f"Score: {best_score}\n")
        file.write("\nРаспознанный текст:\n")
        file.write(best_text)
        file.write("\n\nРазделенные строки:\n")
        for i, line in enumerate(lines):
            file.write(f"{i:2d}: {line}\n")
    
    # Эталонные строки
    expected_lines = []
    expected_lines.append("Задание указано на доске. Подписывать бланк НЕ требуется")
    
    for _ in range(5):
        expected_lines.append("АБВГДЕЁЖЗ ИЙКЛМНОПРСТ У")
        expected_lines.append("ФХЦЧШЩЪЫЬЭЮЯ-0123456789")
    
    for _ in range(5):
        expected_lines.append("АБВГДЕЁЖЗИЙ КЛМНОПРСТ УФХЦЧШЩЪЫЬЭ")
        expected_lines.append("ЮЯ-0123456789")
    
    # Вычисление метрик
    acc_list, cer_list, wer_list = [], [], []
    line_results = []
    
    # Сопоставляем строки
    if len(lines) < len(expected_lines):
        print(f"\nПредупреждение: распознано {len(lines)} строк, ожидается {len(expected_lines)}")
        lines.extend([""] * (len(expected_lines) - len(lines)))
    
    min_lines = min(len(expected_lines), len(lines))
    
    if min_lines == 0:
        print(f"Ошибка: Нет строк для сравнения")
        return None
    
    print("\nСравнение с эталоном:")
    for i in range(min_lines):
        gt = expected_lines[i]
        pred = lines[i] if i < len(lines) else ""
        acc = char_accuracy(gt, pred)
        cer_val = cer(gt, pred)
        wer_val = wer(gt, pred)
        
        acc_list.append(acc)
        cer_list.append(cer_val)
        wer_list.append(wer_val)
        
        line_results.append({
            'file': os.path.basename(image_path),
            'line_num': i,
            'gt': gt,
            'pred': pred,
            'accuracy': acc,
            'cer': cer_val,
            'wer': wer_val
        })
        
        print(f"Строка {i:2d} | ACC: {acc:.3f} | CER: {cer_val:.3f} | WER: {wer_val:.3f}")
        if pred and cer_val > 0.8:
            print(f"    GT: {gt}")
            print(f"    Pred: {pred}")
    
    # Вычисляем средние метрики
    if acc_list:
        file_metrics = {
            'file': os.path.basename(image_path),
            'avg_accuracy': np.mean(acc_list),
            'avg_cer': np.mean(cer_list),
            'avg_wer': np.mean(wer_list),
            'total_lines': min_lines,
            'matched_lines': sum(1 for i in range(min_lines) if cer_list[i] < 0.8),
            'best_method': best_method,
            'text_length': len(best_text),
            'line_results': line_results
        }
        
        print(f"\nИтоги для файла {os.path.basename(image_path)}:")
        print(f"Метод: {best_method}")
        print(f"Длина текста: {len(best_text)} символов")
        print(f"Средняя Accuracy: {file_metrics['avg_accuracy']:.4f}")
        print(f"Средний CER: {file_metrics['avg_cer']:.4f}")
        print(f"Средний WER: {file_metrics['avg_wer']:.4f}")
        print(f"Обработано строк: {file_metrics['total_lines']}")
        
        return file_metrics
    else:
        print(f"\nНе удалось вычислить метрики для файла {image_path}")
        return None

def main():
    # Получаем список файлов
    pattern = os.path.join(FOLDER_PATH, "*_*.png")
    image_files = glob.glob(pattern)
    
    # Фильтруем файлы
    filtered_files = []
    for file in image_files:
        basename = os.path.basename(file)
        name_without_ext = os.path.splitext(basename)[0]
        parts = name_without_ext.split('_')
        
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            filtered_files.append(file)
    
    print(f"Найдено файлов для обработки: {len(filtered_files)}")
    
    if not filtered_files:
        print(f"Файлы с паттерном X_Y.png не найдены")
        return
    
    # Сортируем файлы
    filtered_files.sort(key=lambda x: (
        int(os.path.splitext(os.path.basename(x))[0].split('_')[0]),
        int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
    ))
    
    # Обрабатываем файлы
    all_metrics = []
    all_line_results = []
    
    for i, image_file in enumerate(filtered_files):
        print(f"\n\n{'#'*80}")
        print(f"Файл {i+1}/{len(filtered_files)}: {os.path.basename(image_file)}")
        print(f"{'#'*80}")
        
        metrics = process_image(image_file, save_intermediate=True)
        if metrics:
            all_metrics.append(metrics)
            all_line_results.extend(metrics['line_results'])
        
        # Для отладки: обработаем только первые N файлов
        if i >= 2:  # Обработать только 3 файла для отладки
            print(f"\nОтладочный режим: обработано {i+1} файлов")
            break
    
    if not all_metrics:
        print("\nНе удалось обработать ни одного файла")
        return
    
    # Выводим статистику
    print("\n" + "="*80)
    print("СТАТИСТИКА")
    print("="*80)
    
    df_metrics = pd.DataFrame([{k: v for k, v in m.items() if k != 'line_results'} for m in all_metrics])
    df_lines = pd.DataFrame(all_line_results)
    
    print("\nМетрики по файлам:")
    print(df_metrics.to_string(index=False))
    
    if not df_metrics.empty:
        print(f"\nСредняя Accuracy: {df_metrics['avg_accuracy'].mean():.4f}")
        print(f"Средний CER: {df_metrics['avg_cer'].mean():.4f}")
        print(f"Средний WER: {df_metrics['avg_wer'].mean():.4f}")
    
    # Сохраняем результаты
    os.makedirs("results_paddle", exist_ok=True)
    
    if not df_metrics.empty:
        df_metrics.to_csv("results_paddle/file_metrics.csv", index=False, encoding='utf-8-sig')
    
    if not df_lines.empty:
        df_lines.to_csv("results_paddle/line_metrics.csv", index=False, encoding='utf-8-sig')
    
    with open("results_paddle/summary.txt", "w", encoding="utf-8") as f:
        f.write("ОТЧЕТ PaddleOCR\n")
        f.write("="*50 + "\n\n")
        f.write(f"Обработано файлов: {len(df_metrics)}\n")
        if not df_metrics.empty:
            f.write(f"Средняя Accuracy: {df_metrics['avg_accuracy'].mean():.4f}\n")
            f.write(f"Средний CER: {df_metrics['avg_cer'].mean():.4f}\n")
            f.write(f"Средний WER: {df_metrics['avg_wer'].mean():.4f}\n")
    
    print("\nОбработка завершена!")

if __name__ == "__main__":
    main()