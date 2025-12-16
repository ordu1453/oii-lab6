try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import pytesseract
    import matplotlib.pyplot as plt
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

    def preprocess_and_show_image(image_path):
        """Функция предобработки изображения с отображением в окне"""
        
        print("1. Открываем исходное изображение...")
        # Открываем исходное изображение
        original_img = Image.open(image_path)
        
        # Создаем фигуру для отображения
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Предобработка изображения для распознавания текста', fontsize=16, fontweight='bold')
        
        # 1. Показываем оригинальное изображение
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Исходное изображение', fontweight='bold')
        axes[0, 0].axis('off')
        
        print("2. Конвертируем в оттенки серого...")
        # 2. Конвертируем в оттенки серого
        img = original_img.convert('L')
        axes[0, 1].imshow(img, cmap='gray')
        axes[0, 1].set_title('Оттенки серого', fontweight='bold')
        axes[0, 1].axis('off')
        
        print("3. Применяем автоконтраст...")
        # 3. Автоконтраст
        img = ImageOps.autocontrast(img, cutoff=2)
        axes[1, 0].imshow(img, cmap='gray')
        axes[1, 0].set_title('Автоконтраст', fontweight='bold')
        axes[1, 0].axis('off')
        
        print("4. Увеличиваем резкость...")
        # 4. Увеличение резкости
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        print("5. Применяем бинаризацию...")
        # 5. Бинаризация (пороговая обработка)
        threshold = 150
        img_binary = img.point(lambda x: 255 if x > threshold else 0)
        axes[1, 1].imshow(img_binary, cmap='gray')
        axes[1, 1].set_title('После бинаризации', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Добавляем информацию о размерах
        for ax, img_display, title in [
            (axes[0, 0], original_img, f"Исходное: {original_img.size}"),
            (axes[0, 1], img, f"Серое: {img.size}"),
            (axes[1, 0], img, f"Контраст: {img.size}"),
            (axes[1, 1], img_binary, f"Бинарное: {img_binary.size}")
        ]:
            ax.text(0.5, -0.1, title, transform=ax.transAxes, 
                   ha='center', fontsize=9, style='italic')
        
        plt.tight_layout()
        print("\nОткрываю окно с результатами предобработки...")
        print("Закройте окно, чтобы продолжить распознавание текста.")
        plt.show()
        
        return img_binary
    
    # Основной процесс
    print("=" * 70)
    print("СИСТЕМА РАСПОЗНАВАНИЯ ТЕКСТА С ВИЗУАЛЬНОЙ ПРЕДОБРАБОТКОЙ")
    print("=" * 70)
    
    image_path = '1/1_Страница_01_Изображение_0001.png'
    
    # Предобработка и отображение
    processed_img = preprocess_and_show_image(image_path)
    
    # Сохраняем обработанное изображение
    processed_img.save('preprocessed_image.png')
    print(f"\nОбработанное изображение сохранено как 'preprocessed_image.png'")
    
    # Распознавание текста
    print("\n" + "=" * 70)
    print("РАСПОЗНАВАНИЕ ТЕКСТА...")
    print("=" * 70)
    
    # Настройки для Tesseract
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    
    # Распознаем текст
    text = pytesseract.image_to_string(processed_img, 
                                      lang='rus', 
                                      config=custom_config)
    
    # Выводим результаты
    print("\nРЕЗУЛЬТАТ РАСПОЗНАВАНИЯ:")
    print("-" * 70)
    print(text)
    print("-" * 70)
    
    # Дополнительная информация
    print(f"\nДополнительная информация:")
    print(f"Длина текста: {len(text)} символов")
    print(f"Количество строк: {len(text.splitlines())}")
    
    # Сохраняем текст в файл
    with open('recognized_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Текст сохранен в файл 'recognized_text.txt'")

except ImportError as e:
    print(f"Ошибка импорта модуля: {e}")
    print("\nУстановите необходимые библиотеки:")
    print("pip install pillow pytesseract matplotlib")
except FileNotFoundError as e:
    print(f"Файл не найден: {e}")
    print("Проверьте путь к файлу изображения или Tesseract OCR")
except Exception as e:
    print(f"Произошла ошибка: {e}")

finally:
    print("\n" + "=" * 70)
    print("Обработка завершена.")
    print("=" * 70)