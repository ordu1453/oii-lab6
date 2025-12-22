import cv2
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher
import os
import glob
import pandas as pd
from paddleocr import PaddleOCR
import json

# Инициализация PaddleOCR
ocr = PaddleOCR(
    text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    device="cpu",  # Измените на "gpu:0" если есть GPU
)

# Папка с изображениями
FOLDER_PATH = "1"

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
    
    # Бинаризуем изображение
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marks = []
    
    # Определяем область поиска для верхней левой метки
    search_area_w = w // 2
    search_area_h = h // 2
    
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
    
    marks.sort(key=lambda m: (m[1], m[0]))
    
    # Выбираем самую верхнюю левую метку
    x, y, mark_w, mark_h, area = marks[0]
    
    # Вычисляем координаты для обрезки
    crop_start_x = x + mark_w + offset
    crop_start_y = y + mark_h + offset
    
    # Обрезаем изображение
    roi = image[crop_start_y:, crop_start_x:]
    
    if roi.size == 0:
        print("ROI пустой, возвращаю оригинальное изображение")
        return original, [(x, y, mark_w, mark_h)]
    
    return roi, [(x, y, mark_w, mark_h)]

def draw_marks(image, marks, color=(0, 255, 0), thickness=2):
    debug_img = image.copy()
    for x, y, w, h in marks:
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, thickness)
    return debug_img

def char_accuracy(gt, pred):
    return SequenceMatcher(None, gt, pred).ratio()

def paddleocr_to_text(result):
    """
    Преобразует результат PaddleOCR в текст с разделением на строки.
    
    Args:
        result: результат от ocr.ocr()
        
    Returns:
        list: список строк текста
    """
    lines = []
    current_y = None
    current_line = []
    
    if result is None or len(result) == 0:
        return lines
    
    # Извлекаем все текстовые блоки
    all_boxes = []
    for line in result:
        boxes = line[0]
        text = line[1][0]
        confidence = line[1][1]
        
        # Вычисляем центральную y-координату bounding box
        y_center = np.mean([point[1] for point in boxes])
        
        all_boxes.append({
            'y_center': y_center,
            'text': text,
            'confidence': confidence,
            'box': boxes
        })
    
    # Сортируем по y-координате
    all_boxes.sort(key=lambda x: x['y_center'])
    
    # Группируем близкие строки
    line_threshold = 20  # Порог для группировки строк
    current_line_items = []
    
    for item in all_boxes:
        if current_y is None:
            current_y = item['y_center']
            current_line_items.append(item)
        elif abs(item['y_center'] - current_y) < line_threshold:
            current_line_items.append(item)
        else:
            # Сортируем элементы в строке по x-координате
            current_line_items.sort(key=lambda x: np.mean([point[0] for point in x['box']]))
            line_text = ' '.join([item['text'] for item in current_line_items])
            lines.append(line_text.strip())
            
            # Начинаем новую строку
            current_line_items = [item]
            current_y = item['y_center']
    
    # Добавляем последнюю строку
    if current_line_items:
        current_line_items.sort(key=lambda x: np.mean([point[0] for point in x['box']]))
        line_text = ' '.join([item['text'] for item in current_line_items])
        lines.append(line_text.strip())
    
    return lines

def process_image(image_path, save_intermediate=False, save_json=False):
    print(f"\n{'='*50}")
    print(f"Обработка файла: {os.path.basename(image_path)}")
    print('='*50)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_path}")
        return None
    
    # Обрезка по меткам
    roi, marks = crop_by_marks(img)
    
    if save_intermediate:
        os.makedirs("debug", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        draw = draw_marks(img, marks)
        cv2.imwrite(f"debug/{base_name}_marks_paddle.png", draw)
    
    if roi is None or roi.size == 0:
        print("Ошибка: ROI пустой, используем оригинальное изображение")
        roi = img.copy()
    
    # Предобработка (опционально, PaddleOCR хорошо работает и без нее)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (9, 9))
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if save_intermediate:
        cv2.imwrite(f"debug/{base_name}_binary_paddle.png", thresh)
        cv2.imwrite(f"debug/{base_name}_roi_paddle.png", roi)
    
    # Сохраняем временное изображение для PaddleOCR
    temp_path = f"temp_{os.path.basename(image_path)}"
    cv2.imwrite(temp_path, roi)
    
    try:
        result = ocr.predict(roi)  
        # for res in result:  
        #     res.print()  
        #     res.save_to_img("output")  
        #     res.save_to_json("output")

        # Извлекаем все распознанные тексты
        all_texts = []
        for res in result:
            if hasattr(res, 'rec_texts'):
                all_texts.extend(res.rec_texts)
            elif 'rec_texts' in res:
                all_texts.extend(res['rec_texts'])

        # Выводим все тексты
        for text in all_texts:
            print(text)

        # Или объединяем в одну строку
        full_text = "\n".join(all_texts)
        print("\nПолный текст:")
        print(full_text)
        
        raw_text = full_text

        # Разбиваем на строки (lines)
        lines = full_text.split('\n')
        
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Сохраняем распознанный текст
    text_file = os.path.splitext(image_path)[0] + "_paddle.txt"
    with open(text_file, "w", encoding="utf-8") as file:
        file.write(raw_text)
    
    print(f"Распознано строк PaddleOCR: {len(lines)}")
    for i, line in enumerate(lines[:5]):  # Показываем первые 5 строк для отладки
        print(f"  Строка {i}: {line}")
    
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
    
    min_lines = min(len(expected_lines), len(lines))
    
    if min_lines == 0:
        print(f"Ошибка: Нет строк для сравнения в файле {image_path}")
        return None
    
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
    
    # Вычисляем средние метрики для файла
    file_metrics = {
        'file': os.path.basename(image_path),
        'avg_accuracy': np.mean(acc_list),
        'avg_cer': np.mean(cer_list),
        'avg_wer': np.mean(wer_list),
        'total_lines': min_lines,
        'matched_lines': sum(1 for i in range(min_lines) if cer_list[i] < 0.5),
        'ocr_engine': 'PaddleOCR',
        'line_results': line_results
    }
    
    print(f"\nИтоги для файла {os.path.basename(image_path)}:")
    print(f"Средняя Accuracy: {file_metrics['avg_accuracy']:.4f}")
    print(f"Средний CER: {file_metrics['avg_cer']:.4f}")
    print(f"Средний WER: {file_metrics['avg_wer']:.4f}")
    print(f"Обработано строк: {file_metrics['total_lines']}")
    
    return file_metrics

def main():
    # Получаем список всех файлов X_Y.png в папке
    pattern = os.path.join(FOLDER_PATH, "*_*.png")
    image_files = glob.glob(pattern)
    
    # Фильтруем только файлы с паттерном X_Y.png (где X и Y - числа)
    filtered_files = []
    for file in image_files:
        basename = os.path.basename(file)
        name_without_ext = os.path.splitext(basename)[0]
        parts = name_without_ext.split('_')
        
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            filtered_files.append(file)
    
    print(f"Найдено файлов для обработки: {len(filtered_files)}")
    
    if not filtered_files:
        print(f"Файлы с паттерном X_Y.png не найдены в папке '{FOLDER_PATH}'")
        return
    
    # Сортируем файлы по X и Y
    filtered_files.sort(key=lambda x: (
        int(os.path.splitext(os.path.basename(x))[0].split('_')[0]),
        int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
    ))
    
    # Обрабатываем все файлы
    all_metrics = []
    all_line_results = []
    
    for image_file in filtered_files:
        metrics = process_image(image_file, save_intermediate=True, save_json=True)
        if metrics:
            all_metrics.append(metrics)
            all_line_results.extend(metrics['line_results'])
    
    if not all_metrics:
        print("Не удалось обработать ни одного файла")
        return
    
    # Выводим общую статистику
    print("\n" + "="*80)
    print("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ФАЙЛАМ (PaddleOCR)")
    print("="*80)
    
    df_metrics = pd.DataFrame([{k: v for k, v in m.items() if k != 'line_results'} for m in all_metrics])
    df_lines = pd.DataFrame(all_line_results)
    
    print("\nМетрики по файлам:")
    print(df_metrics.to_string(index=False))
    
    print("\n" + "="*80)
    print("СРЕДНИЕ МЕТРИКИ ПО ВСЕМ ФАЙЛАМ:")
    print("="*80)
    print(f"Средняя Accuracy по всем файлам: {df_metrics['avg_accuracy'].mean():.4f}")
    print(f"Средний CER по всем файлам: {df_metrics['avg_cer'].mean():.4f}")
    print(f"Средний WER по всем файлам: {df_metrics['avg_wer'].mean():.4f}")
    print(f"Общее количество обработанных файлов: {len(df_metrics)}")
    print(f"Общее количество обработанных строк: {len(df_lines)}")
    
    print("\n" + "="*80)
    print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:")
    print("="*80)
    
    accuracy_stats = df_lines['accuracy'].describe()
    cer_stats = df_lines['cer'].describe()
    wer_stats = df_lines['wer'].describe()
    
    print(f"\nТочность (Accuracy) по строкам:")
    print(f"  Среднее: {accuracy_stats['mean']:.4f}")
    print(f"  Медиана: {accuracy_stats['50%']:.4f}")
    print(f"  Стандартное отклонение: {accuracy_stats['std']:.4f}")
    print(f"  Минимум: {accuracy_stats['min']:.4f}")
    print(f"  Максимум: {accuracy_stats['max']:.4f}")
    
    print(f"\nCER по строкам:")
    print(f"  Среднее: {cer_stats['mean']:.4f}")
    print(f"  Медиана: {cer_stats['50%']:.4f}")
    print(f"  Стандартное отклонение: {cer_stats['std']:.4f}")
    
    print(f"\nWER по строкам:")
    print(f"  Среднее: {wer_stats['mean']:.4f}")
    print(f"  Медиана: {wer_stats['50%']:.4f}")
    print(f"  Стандартное отклонение: {wer_stats['std']:.4f}")
    
    print("\nРаспределение строк по уровню CER:")
    bins = [0, 0.1, 0.2, 0.3, 0.5, 1.0, float('inf')]
    labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.5', '0.5-1.0', '>1.0']
    df_lines['cer_bin'] = pd.cut(df_lines['cer'], bins=bins, labels=labels, right=False)
    cer_distribution = df_lines['cer_bin'].value_counts().sort_index()
    
    for bin_label, count in cer_distribution.items():
        percentage = (count / len(df_lines)) * 100
        print(f"  CER {bin_label}: {count} строк ({percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("="*80)
    
    os.makedirs("results", exist_ok=True)
    
    df_metrics.to_csv("results/file_metrics_paddle.csv", index=False, encoding='utf-8-sig')
    df_lines.to_csv("results/line_metrics_paddle.csv", index=False, encoding='utf-8-sig')
    
    with open("results/summary_report_paddle.txt", "w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ПО ОБРАБОТКЕ OCR (PaddleOCR)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Обработано файлов: {len(df_metrics)}\n")
        f.write(f"Обработано строк: {len(df_lines)}\n\n")
        
        f.write("СРЕДНИЕ МЕТРИКИ:\n")
        f.write(f"Средняя Accuracy: {df_metrics['avg_accuracy'].mean():.4f}\n")
        f.write(f"Средний CER: {df_metrics['avg_cer'].mean():.4f}\n")
        f.write(f"Средний WER: {df_metrics['avg_wer'].mean():.4f}\n\n")
        
        f.write("МЕТРИКИ ПО ФАЙЛАМ:\n")
        f.write(df_metrics.to_string(index=False))
    
    print("\nОбработка завершена успешно!")
    print(f"Результаты сохранены в папке 'results' с суффиксом '_paddle'")

if __name__ == "__main__":
    main()