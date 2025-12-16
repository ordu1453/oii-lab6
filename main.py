# main_fixed.py
import os
import sys
import warnings
import zipfile
import tempfile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Отключаем предупреждения NumPy
warnings.filterwarnings('ignore')

# Проверяем версию NumPy
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    # Для NumPy 2.x используем новые функции если нужно
    if hasattr(np, 'float128'):
        np.float128 = np.longdouble  # Совместимость для старых кодов
except ImportError as e:
    print(f"Ошибка импорта NumPy: {e}")
    print("Установите NumPy: pip install numpy==2.0.0")
    sys.exit(1)

# Импорт остальных библиотек с обработкой ошибок
try:
    from PIL import Image
    import pytesseract
    from pdf2image import convert_from_path
    import easyocr
except ImportError as e:
    print(f"Ошибка импорта библиотек: {e}")
    print("Установите зависимости: pip install Pillow pytesseract pdf2image easyocr")
    sys.exit(1)

class OCRComparator:
    def __init__(self, zip_path):
        """Инициализация OCR компаратора"""
        self.zip_path = zip_path
        self.temp_dir = tempfile.mkdtemp()
        self.results = []
        
        # Настройка пути к Tesseract для Windows
        # Раскомментируйте и укажите правильный путь:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Инициализация EasyOCR
        try:
            self.easyocr_reader = easyocr.Reader(['ru'], gpu=False)
            print("EasyOCR инициализирован")
        except Exception as e:
            print(f"Ошибка инициализации EasyOCR: {e}")
            self.easyocr_reader = None
        
        # Шаблон для сравнения
        self.template_text = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789"
        print(f"Шаблон содержит: {len(self.template_text)} символов")
    
    def extract_pdfs(self):
        """Извлечение PDF файлов из архива"""
        print(f"Извлечение файлов из {self.zip_path}")
        
        if not os.path.exists(self.zip_path):
            print(f"Файл {self.zip_path} не найден!")
            return []
        
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            
            pdf_files = list(Path(self.temp_dir).glob("*.pdf"))
            print(f"Найдено {len(pdf_files)} PDF файлов")
            return sorted(pdf_files, key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
        
        except Exception as e:
            print(f"Ошибка при извлечении: {e}")
            return []
    
    def convert_pdf_to_image(self, pdf_path):
        """Конвертация PDF в изображение"""
        try:
            images = convert_from_path(pdf_path, dpi=150)
            return images[0] if images else None
        except Exception as e:
            print(f"Ошибка конвертации {pdf_path}: {e}")
            return None
    
    def preprocess_image(self, image):
        """Предобработка изображения"""
        try:
            # Простая предобработка
            # Конвертируем в оттенки серого
            if image.mode != 'L':
                image = image.convert('L')
            
            # Увеличиваем контраст
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            return image
        
        except Exception as e:
            print(f"Ошибка предобработки: {e}")
            return image
    
    def ocr_with_tesseract(self, image):
        """Распознавание с помощью Tesseract"""
        try:
            text = pytesseract.image_to_string(
                image, 
                lang='rus+eng',
                config='--psm 6 --oem 3'
            )
            return text.strip()
        except Exception as e:
            print(f"Ошибка Tesseract: {e}")
            return ""
    
    def ocr_with_easyocr(self, image):
        """Распознавание с помощью EasyOCR"""
        if not self.easyocr_reader:
            return ""
        
        try:
            # Конвертируем PIL Image в numpy array
            import numpy as np
            img_array = np.array(image)
            
            # Если изображение в оттенках серого, конвертируем в RGB
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array]*3, axis=-1)
            
            # Распознавание
            results = self.easyocr_reader.readtext(img_array, detail=0)
            return " ".join(results).strip()
        
        except Exception as e:
            print(f"Ошибка EasyOCR: {e}")
            return ""
    
    def calculate_similarity(self, text1, text2):
        """Вычисление схожести двух текстов"""
        # Приводим к верхнему регистру и удаляем пробелы
        t1 = "".join(text1.upper().split())
        t2 = "".join(text2.upper().split())
        
        if not t2:
            return 0.0
        
        # Считаем совпадающие символы
        matches = 0
        for c1, c2 in zip(t1, t2):
            if c1 == c2:
                matches += 1
        
        return (matches / len(t2)) * 100
    
    def process_files(self, max_files=3):
        """Обработка файлов"""
        pdf_files = self.extract_pdfs()
        
        if not pdf_files:
            print("Не найдено PDF файлов для обработки")
            return
        
        print(f"\nНачинаю обработку {min(max_files, len(pdf_files))} файлов...")
        
        for i, pdf_path in enumerate(pdf_files[:max_files]):
            print(f"\n[{i+1}/{min(max_files, len(pdf_files))}] Файл: {pdf_path.name}")
            
            # Конвертируем PDF в изображение
            image = self.convert_pdf_to_image(pdf_path)
            if not image:
                continue
            
            # Предобработка
            processed_image = self.preprocess_image(image)
            
            # Распознавание Tesseract
            print("  Tesseract распознавание...")
            text_tesseract = self.ocr_with_tesseract(processed_image)
            accuracy_tesseract = self.calculate_similarity(text_tesseract, self.template_text)
            
            # Распознавание EasyOCR
            print("  EasyOCR распознавание...")
            text_easyocr = self.ocr_with_easyocr(image)  # Используем оригинальное изображение
            accuracy_easyocr = self.calculate_similarity(text_easyocr, self.template_text)
            
            # Сохраняем результаты
            self.results.append({
                'file': pdf_path.name,
                'tesseract_text': text_tesseract[:100] + "..." if len(text_tesseract) > 100 else text_tesseract,
                'tesseract_accuracy': round(accuracy_tesseract, 2),
                'easyocr_text': text_easyocr[:100] + "..." if len(text_easyocr) > 100 else text_easyocr,
                'easyocr_accuracy': round(accuracy_easyocr, 2)
            })
            
            print(f"    Tesseract: {accuracy_tesseract:.1f}%")
            print(f"    EasyOCR: {accuracy_easyocr:.1f}%")
            
            # Сохраняем изображение для отладки
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            processed_image.save(debug_dir / f"{pdf_path.stem}_processed.png")
    
    def save_results(self):
        """Сохранение результатов"""
        if not self.results:
            print("Нет результатов для сохранения")
            return
        
        # Создаем DataFrame
        df = pd.DataFrame(self.results)
        
        # Создаем папку для результатов
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Сохраняем в разных форматах
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV
        csv_path = results_dir / f"ocr_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Excel
        excel_path = results_dir / f"ocr_results_{timestamp}.xlsx"
        df.to_excel(excel_path, index=False)
        
        print(f"\nРезультаты сохранены:")
        print(f"  CSV: {csv_path}")
        print(f"  Excel: {excel_path}")
        
        # Выводим сводку
        print(f"\nСводка результатов:")
        print(df[['file', 'tesseract_accuracy', 'easyocr_accuracy']].to_string())
        
        return df
    
    def create_visualization(self, df):
        """Создание визуализаций"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # График 1: Сравнение точности
            ax1 = axes[0]
            x = range(len(df))
            width = 0.35
            
            ax1.bar([i - width/2 for i in x], df['tesseract_accuracy'], 
                   width, label='Tesseract', alpha=0.8, color='blue')
            ax1.bar([i + width/2 for i in x], df['easyocr_accuracy'], 
                   width, label='EasyOCR', alpha=0.8, color='green')
            
            ax1.set_xlabel('Файлы')
            ax1.set_ylabel('Точность (%)')
            ax1.set_title('Сравнение точности OCR')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Средняя точность
            ax2 = axes[1]
            avg_acc = [df['tesseract_accuracy'].mean(), df['easyocr_accuracy'].mean()]
            bars = ax2.bar(['Tesseract', 'EasyOCR'], avg_acc, 
                          color=['blue', 'green'])
            
            ax2.set_ylabel('Средняя точность (%)')
            ax2.set_title('Средняя точность распознавания')
            ax2.grid(True, alpha=0.3)
            
            # Добавляем значения на столбцы
            for bar, acc in zip(bars, avg_acc):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{acc:.1f}%', ha='center', va='bottom')
            
            plt.suptitle('Сравнение OCR библиотек для рукописного текста', fontsize=14)
            plt.tight_layout()
            
            # Сохраняем график
            plot_path = Path("results") / f"ocr_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\nГрафик сохранен: {plot_path}")
            
        except Exception as e:
            print(f"Ошибка при создании графиков: {e}")
    
    def cleanup(self):
        """Очистка временных файлов"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            print(f"Временные файлы очищены")
        except Exception as e:
            print(f"Ошибка при очистке: {e}")

def main():
    """Основная функция"""
    print("="*60)
    print("Сравнение OCR библиотек для рукописного текста")
    print("="*60)
    
    # Путь к архиву
    zip_path = "Тест для французских булочек и_чаю.zip"
    
    # Проверка наличия файла
    if not os.path.exists(zip_path):
        print(f"Файл {zip_path} не найден!")
        print("Поместите архив в ту же папку, что и скрипт.")
        return
    
    # Создаем компаратор
    comparator = OCRComparator(zip_path)
    
    try:
        # Обрабатываем файлы
        comparator.process_files(max_files=5)
        
        # Сохраняем результаты
        df = comparator.save_results()
        
        # Создаем визуализацию
        if df is not None and not df.empty:
            comparator.create_visualization(df)
    
    except Exception as e:
        print(f"Ошибка при выполнении: {e}")
    
    finally:
        # Очищаем временные файлы
        comparator.cleanup()
    
    print("\n" + "="*60)
    print("Обработка завершена!")
    print("="*60)

if __name__ == "__main__":
    main()