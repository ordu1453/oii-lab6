
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
import argparse
import os
from typing import Dict, List, Tuple, Optional
import json

# Установите путь к tesseract, если он не добавлен в PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # для Linux/Mac


class ICRProcessor:
    def __init__(self, lang='rus', config='--oem 3 --psm 6'):
        """
        Инициализация процессора ICR
        lang: язык для распознавания ('rus' для русского)
        config: конфигурация Tesseract
        """
        self.lang = lang
        self.config = config
        
        # Шаблон символов для проверки (русские буквы и цифры)
        self.expected_chars = {
            'letters': 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
            'digits': '0123456789',
            'all': 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789'
        }
        
    def preprocess_image(self, image_path):
        """
        Предобработка изображения для улучшения распознавания
        """
        # Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        # Конвертация в оттенки серого
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Увеличение контраста с помощью CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        
        # Применение пороговой обработки (бинаризация)
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Удаление шума
        denoised = cv2.medianBlur(binary, 3)
        
        # Увеличение резкости
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened
    
    def extract_text(self, image_path, preprocess=True):
        """
        Извлечение текста из изображения
        """
        if preprocess:
            processed_img = self.preprocess_image(image_path)
        else:
            processed_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
        # Преобразование в PIL Image для tesseract
        pil_image = Image.fromarray(processed_img)
        
        # Распознавание текста
        text = pytesseract.image_to_string(
            pil_image, 
            lang=self.lang, 
            config=self.config
        )
        
        return text
    
    def extract_text_with_confidence(self, image_path, preprocess=True):
        """
        Извлечение текста с информацией о уверенности распознавания
        """
        if preprocess:
            processed_img = self.preprocess_image(image_path)
        else:
            processed_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
        pil_image = Image.fromarray(processed_img)
        
        # Получение подробных данных от Tesseract
        data = pytesseract.image_to_data(
            pil_image, 
            lang=self.lang, 
            config=self.config,
            output_type=pytesseract.Output.DICT
        )
        
        return data
    
    def analyze_recognition_accuracy(self, recognized_text: str, expected_text: str) -> Dict:
        """
        Анализ точности распознавания по сравнению с ожидаемым текстом
        """
        # Очистка текста от лишних пробелов и переносов строк
        recognized_clean = recognized_text.replace('\n', '').replace(' ', '').strip()
        expected_clean = expected_text.replace('\n', '').replace(' ', '').strip()
        
        # Выравнивание строк по длине (добавление пробелов если нужно)
        max_len = max(len(recognized_clean), len(expected_clean))
        recognized_padded = recognized_clean.ljust(max_len)
        expected_padded = expected_clean.ljust(max_len)
        
        # Подсчет правильных, неправильных и пропущенных символов
        correct = 0
        incorrect = 0
        missing = 0
        extra = 0
        
        # Для детального сравнения
        comparison = []
        
        for i in range(max_len):
            rec_char = recognized_padded[i] if i < len(recognized_clean) else ''
            exp_char = expected_padded[i] if i < len(expected_clean) else ''
            
            if exp_char == ' ' or exp_char == '':
                if rec_char != '':
                    extra += 1
                    comparison.append({'position': i, 'expected': exp_char, 
                                     'recognized': rec_char, 'status': 'extra'})
                continue
                
            if rec_char == '':
                missing += 1
                comparison.append({'position': i, 'expected': exp_char, 
                                 'recognized': rec_char, 'status': 'missing'})
            elif rec_char == exp_char:
                correct += 1
                comparison.append({'position': i, 'expected': exp_char, 
                                 'recognized': rec_char, 'status': 'correct'})
            else:
                incorrect += 1
                comparison.append({'position': i, 'expected': exp_char, 
                                 'recognized': rec_char, 'status': 'incorrect'})
        
        # Расчет метрик
        total_expected = len(expected_clean.replace(' ', ''))
        if total_expected > 0:
            accuracy = (correct / total_expected) * 100
        else:
            accuracy = 0
            
        return {
            'total_expected': total_expected,
            'correct': correct,
            'incorrect': incorrect,
            'missing': missing,
            'extra': extra,
            'accuracy': accuracy,
            'comparison': comparison,
            'recognized_text': recognized_text,
            'expected_text': expected_text
        }
    
    def process_icr_template(self, image_path, template_type='alphabet_digits', 
                           rows=5, chars_per_row=None):
        """
        Обработка ICR бланка с шаблоном символов
        template_type: 'alphabet_digits', 'letters_only', 'digits_only'
        rows: количество строк с символами
        chars_per_row: количество символов в каждой строке
        """
        # Получение текста с уверенностью
        data = self.extract_text_with_confidence(image_path)
        recognized_text = ' '.join([word for word in data['text'] if word.strip()])
        
        # Генерация ожидаемого текста на основе шаблона
        if template_type == 'alphabet_digits':
            template = self.expected_chars['all']
        elif template_type == 'letters_only':
            template = self.expected_chars['letters']
        elif template_type == 'digits_only':
            template = self.expected_chars['digits']
        else:
            template = self.expected_chars['all']
        
        # Если не указано chars_per_row, берем всю длину шаблона
        if chars_per_row is None:
            chars_per_row = len(template)
        
        # Создание ожидаемого текста
        expected_lines = []
        for row in range(rows):
            start_idx = (row * chars_per_row) % len(template)
            line = template[start_idx:start_idx + chars_per_row]
            if len(line) < chars_per_row:
                line += template[:chars_per_row - len(line)]
            expected_lines.append(line)
        
        expected_text = '\n'.join(expected_lines)
        
        # Анализ точности
        accuracy_data = self.analyze_recognition_accuracy(recognized_text, expected_text)
        
        # Дополнительная статистика по типам символов
        letters_accuracy = self._calculate_char_type_accuracy(
            recognized_text, 
            self.expected_chars['letters']
        )
        digits_accuracy = self._calculate_char_type_accuracy(
            recognized_text, 
            self.expected_chars['digits']
        )
        
        accuracy_data['letters_accuracy'] = letters_accuracy
        accuracy_data['digits_accuracy'] = digits_accuracy
        
        return accuracy_data
    
    def _calculate_char_type_accuracy(self, recognized_text: str, char_set: str) -> Dict:
        """
        Расчет точности для конкретного набора символов
        """
        recognized_chars = [c for c in recognized_text if c in char_set]
        expected_chars = char_set
        
        correct = 0
        total = len(expected_chars)
        
        for i, expected_char in enumerate(expected_chars):
            if i < len(recognized_chars) and recognized_chars[i] == expected_char:
                correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'recognized_chars': ''.join(recognized_chars[:total]),
            'expected_chars': expected_chars
        }
    
    def visualize_comparison(self, accuracy_data: Dict, max_chars_per_line=50):
        """
        Визуализация сравнения ожидаемого и распознанного текста
        """
        print("\n" + "="*80)
        print("АНАЛИЗ ТОЧНОСТИ РАСПОЗНАВАНИЯ")
        print("="*80)
        
        print(f"\nОбщая точность: {accuracy_data['accuracy']:.2f}%")
        print(f"Правильно распознано: {accuracy_data['correct']} из {accuracy_data['total_expected']}")
        print(f"Ошибки: {accuracy_data['incorrect']}, Пропущено: {accuracy_data['missing']}, Лишние: {accuracy_data['extra']}")
        
        if 'letters_accuracy' in accuracy_data:
            print(f"\nТочность для букв: {accuracy_data['letters_accuracy']['accuracy']:.2f}% "
                  f"({accuracy_data['letters_accuracy']['correct']}/{accuracy_data['letters_accuracy']['total']})")
        
        if 'digits_accuracy' in accuracy_data:
            print(f"Точность для цифр: {accuracy_data['digits_accuracy']['accuracy']:.2f}% "
                  f"({accuracy_data['digits_accuracy']['correct']}/{accuracy_data['digits_accuracy']['total']})")
        
        print("\n" + "-"*80)
        print("ПОДРОБНОЕ СРАВНЕНИЕ:")
        print("-"*80)
        
        # Разбиваем на строки для лучшего отображения
        expected_lines = accuracy_data['expected_text'].split('\n')
        recognized_lines = accuracy_data['recognized_text'].split('\n')
        
        for i, (exp_line, rec_line) in enumerate(zip(expected_lines, recognized_lines)):
            if i >= len(recognized_lines):
                rec_line = ""
            
            print(f"\nСтрока {i+1}:")
            print(f"Ожидалось:  {exp_line}")
            print(f"Распознано: {rec_line}")
            
            # Показываем различия в компактном формате
            diff_line = ""
            for j in range(min(len(exp_line), len(rec_line))):
                if exp_line[j] == rec_line[j]:
                    diff_line += "✓"
                else:
                    diff_line += "✗"
            
            if diff_line:
                print(f"Совпадения:  {diff_line}")
        
        # Вывод статистики по позициям ошибок
        print("\n" + "-"*80)
        print("АНАЛИЗ ОШИБОК ПО ПОЗИЦИЯМ:")
        print("-"*80)
        
        error_positions = [c for c in accuracy_data['comparison'] 
                          if c['status'] in ['incorrect', 'missing']]
        
        if error_positions:
            for error in error_positions[:20]:  # Показываем первые 20 ошибок
                status_symbol = "✗" if error['status'] == 'incorrect' else "∅"
                print(f"Позиция {error['position']:3d}: {status_symbol} "
                      f"Ожидалось: '{error['expected']}' "
                      f"Распознано: '{error['recognized']}'")
            
            if len(error_positions) > 20:
                print(f"... и еще {len(error_positions) - 20} ошибок")
        else:
            print("Ошибок не обнаружено!")
    
    def generate_accuracy_report(self, accuracy_data: Dict, output_file: str = None):
        """
        Генерация отчета о точности распознавания
        """
        report = {
            'timestamp': np.datetime64('now').astype(str),
            'overall_accuracy': accuracy_data['accuracy'],
            'statistics': {
                'total_expected': accuracy_data['total_expected'],
                'correct': accuracy_data['correct'],
                'incorrect': accuracy_data['incorrect'],
                'missing': accuracy_data['missing'],
                'extra': accuracy_data['extra']
            },
            'detailed_comparison': accuracy_data['comparison']
        }
        
        if 'letters_accuracy' in accuracy_data:
            report['letters_accuracy'] = accuracy_data['letters_accuracy']
        
        if 'digits_accuracy' in accuracy_data:
            report['digits_accuracy'] = accuracy_data['digits_accuracy']
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\nОтчет сохранен в: {output_file}")
        
        return report
    
    def extract_text_from_roi(self, image_path, roi_coords):
        """
        Извлечение текста из определенной области изображения (ROI)
        roi_coords: (x, y, width, height)
        """
        # Загрузка изображения
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        # Выделение ROI
        x, y, w, h = roi_coords
        roi = img[y:y+h, x:x+w]
        
        # Предобработка ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Распознавание текста из ROI
        pil_roi = Image.fromarray(binary)
        text = pytesseract.image_to_string(pil_roi, lang=self.lang, config='--oem 3 --psm 7')
        
        return text.strip()


def main():
    parser = argparse.ArgumentParser(description='Распознавание текста на ICR-бланках с анализом точности')
    parser.add_argument('image_path', help='Путь к изображению бланка')
    parser.add_argument('--lang', default='rus', help='Язык для распознавания (по умолчанию: rus)')
    parser.add_argument('--template', default='alphabet_digits', 
                       choices=['alphabet_digits', 'letters_only', 'digits_only'],
                       help='Тип шаблона бланка')
    parser.add_argument('--rows', type=int, default=5, help='Количество строк на бланке')
    parser.add_argument('--chars_per_row', type=int, default=None, 
                       help='Количество символов в строке (по умолчанию: длина шаблона)')
    parser.add_argument('--output', default='icr_results.txt', help='Файл для сохранения текста')
    parser.add_argument('--report', default='accuracy_report.json', 
                       help='Файл для сохранения отчета о точности')
    parser.add_argument('--visualize', action='store_true', 
                       help='Визуализировать сравнение распознанного и ожидаемого текста')
    
    args = parser.parse_args()
    
    # Проверка существования файла
    if not os.path.exists(args.image_path):
        print(f"Файл не найден: {args.image_path}")
        return
    
    # Создание процессора ICR
    processor = ICRProcessor(lang=args.lang)
    
    try:
        print(f"Обработка ICR бланка: {args.image_path}")
        print(f"Шаблон: {args.template}, Строк: {args.rows}")
        
        # Обработка по шаблону
        accuracy_data = processor.process_icr_template(
            args.image_path,
            template_type=args.template,
            rows=args.rows,
            chars_per_row=args.chars_per_row
        )
        
        # Сохранение распознанного текста
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(accuracy_data['recognized_text'])
        print(f"\nРаспознанный текст сохранен в: {args.output}")
        
        # Визуализация результатов
        if args.visualize:
            processor.visualize_comparison(accuracy_data)
        else:
            print(f"\nОбщая точность: {accuracy_data['accuracy']:.2f}%")
            print(f"Правильно: {accuracy_data['correct']}/{accuracy_data['total_expected']}")
        
        # Генерация отчета
        report = processor.generate_accuracy_report(accuracy_data, args.report)
        
        # Дополнительная информация
        print("\n" + "="*50)
        print("ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        
        # Пример поиска чекбоксов
        print("\nПоиск чекбоксов...")
        img = cv2.imread(args.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        checkbox_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 1000:  # Размер чекбокса
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 0.8 < aspect_ratio < 1.2:  # Примерно квадратный
                    checkbox_count += 1
        
        print(f"Найдено потенциальных чекбоксов: {checkbox_count}")
        
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()