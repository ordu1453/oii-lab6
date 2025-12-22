import cv2
import numpy as np
from jiwer import cer, wer
from difflib import SequenceMatcher
import os
import glob
import pandas as pd
import easyocr
import warnings
from typing import List, Tuple

# Игнорируем предупреждения от EasyOCR
warnings.filterwarnings('ignore')

# Инициализация EasyOCR для русского языка
reader = easyocr.Reader(['ru'], gpu=False)  # Установите gpu=True если есть поддержка CUDA

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

def extract_text_lines_with_easyocr(image) -> List[str]:
    """
    Извлекает текст из изображения с помощью EasyOCR, сохраняя построчную структуру.
    
    Args:
        image: изображение в формате BGR
        
    Returns:
        List[str]: список строк распознанного текста
    """
    # Получаем детальные результаты от EasyOCR (с координатами)
    results = reader.readtext(image, paragraph=False, detail=1)
    
    # Если результатов нет, возвращаем пустой список
    if not results:
        return []
    
    # Группируем текст по строкам на основе Y-координаты bounding box
    lines_dict = {}
    
    for bbox, text, confidence in results:
        # Вычисляем среднюю Y-координату bounding box
        y_coords = [point[1] for point in bbox]
        avg_y = sum(y_coords) / len(y_coords)
        
        # Находим ближайшую группу строк (в пределах порога)
        found_group = None
        for group_y in lines_dict.keys():
            if abs(avg_y - group_y) < 15:  # Порог для объединения в одну строку
                found_group = group_y
                break
        
        if found_group is not None:
            # Добавляем к существующей строке
            lines_dict[found_group].append((bbox, text, avg_y))
        else:
            # Создаем новую строку
            lines_dict[avg_y] = [(bbox, text, avg_y)]
    
    # Сортируем строки по Y-координате (сверху вниз)
    sorted_groups = sorted(lines_dict.items(), key=lambda x: x[0])
    
    # Для каждой группы сортируем слова по X-координате (слева направо)
    lines = []
    for group_y, elements in sorted_groups:
        # Сортируем элементы в строке по X-координате
        elements_sorted = sorted(elements, key=lambda elem: min(point[0] for point in elem[0]))
        
        # Объединяем слова в строку
        line_text = ' '.join([text for _, text, _ in elements_sorted])
        lines.append(line_text.strip())
    
    return lines

def process_image(image_path, save_intermediate=False):
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
        cv2.imwrite(f"debug/{base_name}_marks_easyocr.png", draw)
    
    if roi is None or roi.size == 0:
        print("Ошибка: ROI пустой, используем оригинальное изображение")
        roi = img.copy()
    
    # Пробуем разные варианты предобработки для лучшего распознавания
    lines_list = []
    
    # Вариант 1: оригинальное изображение
    print("Распознавание на оригинальном изображении...")
    lines1 = extract_text_lines_with_easyocr(roi)
    lines_list.append(("original", lines1))
    
    # Вариант 2: предобработанное бинарное изображение
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))  # Уменьшаем blur для сохранения деталей
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    print("Распознавание на бинарном изображении...")
    lines2 = extract_text_lines_with_easyocr(processed_img)
    lines_list.append(("binary", lines2))
    
    # Вариант 3: адаптивная бинаризация
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    adaptive_img = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    
    print("Распознавание на адаптивно бинаризованном изображении...")
    lines3 = extract_text_lines_with_easyocr(adaptive_img)
    lines_list.append(("adaptive", lines3))
    
    # Выбираем вариант с максимальным количеством строк
    best_variant = max(lines_list, key=lambda x: len(x[1]))
    print(f"Выбран вариант: {best_variant[0]} с {len(best_variant[1])} строками")
    
    lines = best_variant[1]
    
    if save_intermediate:
        cv2.imwrite(f"debug/{base_name}_binary_easyocr.png", thresh)
        cv2.imwrite(f"debug/{base_name}_adaptive_easyocr.png", adaptive)
        cv2.imwrite(f"debug/{base_name}_roi_easyocr.png", roi)
        
        # Сохраняем изображение с bounding boxes для отладки
        debug_img = roi.copy()
        results = reader.readtext(roi, paragraph=False, detail=1)
        for bbox, text, confidence in results:
            # Конвертируем bounding box в нужный формат
            bbox = np.array(bbox, dtype=np.int32)
            cv2.polylines(debug_img, [bbox], True, (0, 255, 0), 2)
            # Добавляем текст
            cv2.putText(debug_img, text, (bbox[0][0], bbox[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(f"debug/{base_name}_bboxes_easyocr.png", debug_img)
    
    # Сохраняем распознанный текст
    raw_text = '\n'.join(lines)
    text_file = os.path.splitext(image_path)[0] + "_easyocr" + ".txt"
    with open(text_file, "w", encoding="utf-8") as file:
        file.write(raw_text)
    
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
    
    # Для каждой ожидаемой строки находим наиболее подходящую распознанную
    for i, gt in enumerate(expected_lines):
        best_match = ""
        best_cer = float('inf')
        
        # Ищем лучшую совпадающую строку среди распознанных
        for pred in lines:
            if pred:  # Пропускаем пустые строки
                current_cer = cer(gt, pred)
                if current_cer < best_cer:
                    best_cer = current_cer
                    best_match = pred
        
        # Если нашли подходящую строку, используем ее
        if best_match and best_cer < 0.8:  # Порог для принятия решения
            pred = best_match
        else:
            pred = ""  # Не нашли подходящую строку
            
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
            'wer': wer_val,
            'matched': 1 if pred else 0
        })
        
        print(f"Строка {i:2d} | ACC: {acc:.3f} | CER: {cer_val:.3f} | WER: {wer_val:.3f}")
        print(f"  Эталон: {gt}")
        print(f"  Распознано: {pred if pred else '(не распознано)'}")
    
    # Вычисляем средние метрики для файла
    file_metrics = {
        'file': os.path.basename(image_path),
        'avg_accuracy': np.mean(acc_list),
        'avg_cer': np.mean(cer_list),
        'avg_wer': np.mean(wer_list),
        'total_lines': len(expected_lines),
        'matched_lines': sum(r['matched'] for r in line_results),
        'line_results': line_results
    }
    
    print(f"\nИтоги для файла {os.path.basename(image_path)}:")
    print(f"Средняя Accuracy: {file_metrics['avg_accuracy']:.4f}")
    print(f"Средний CER: {file_metrics['avg_cer']:.4f}")
    print(f"Средний WER: {file_metrics['avg_wer']:.4f}")
    print(f"Ожидалось строк: {file_metrics['total_lines']}")
    print(f"Совпало строк: {file_metrics['matched_lines']}")
    print(f"Всего распознано строк EasyOCR: {len(lines)}")
    
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
    
    for i, image_file in enumerate(filtered_files, 1):
        print(f"\n{'#'*60}")
        print(f"Обработка файла {i}/{len(filtered_files)}")
        metrics = process_image(image_file, save_intermediate=True)
        if metrics:
            all_metrics.append(metrics)
            all_line_results.extend(metrics['line_results'])
    
    if not all_metrics:
        print("Не удалось обработать ни одного файла")
        return
    
    # Выводим общую статистику
    print("\n" + "="*80)
    print("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ ФАЙЛАМ")
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
    
    print(f"\nПроцент совпадения строк: {(df_lines['matched'].mean() * 100):.1f}%")
    
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
    
    df_metrics.to_csv("results/file_metrics_easyocr.csv", index=False, encoding='utf-8-sig')
    df_lines.to_csv("results/line_metrics_easyocr.csv", index=False, encoding='utf-8-sig')
    
    with open("results/summary_report_easyocr.txt", "w", encoding="utf-8") as f:
        f.write("ОТЧЕТ ПО ОБРАБОТКЕ OCR (EasyOCR)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Обработано файлов: {len(df_metrics)}\n")
        f.write(f"Обработано строк: {len(df_lines)}\n")
        f.write(f"Процент совпадения строк: {(df_lines['matched'].mean() * 100):.1f}%\n\n")
        
        f.write("СРЕДНИЕ МЕТРИКИ:\n")
        f.write(f"Средняя Accuracy: {df_metrics['avg_accuracy'].mean():.4f}\n")
        f.write(f"Средний CER: {df_metrics['avg_cer'].mean():.4f}\n")
        f.write(f"Средний WER: {df_metrics['avg_wer'].mean():.4f}\n\n")
        
        f.write("МЕТРИКИ ПО ФАЙЛАМ:\n")
        f.write(df_metrics.to_string(index=False))
    
    print("\nОбработка завершена успешно!")
    print("Результаты сохранены в папке 'results' с суффиксом '_easyocr'")

if __name__ == "__main__":
    main()