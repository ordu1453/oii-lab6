try:
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract
    import numpy as np

    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

    def preprocess_image(image_path):
        """Функция предобработки изображения для улучшения распознавания текста"""
        
        # Открываем изображение
        img = Image.open(image_path)
        
        # 1. Конвертируем в оттенки серого
        img = img.convert('L')
        
        # 2. Повышаем контрастность
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)  # Увеличиваем контрастность в 2 раза
        
        # 3. Повышаем резкость
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  # Увеличиваем резкость в 2 раза
        
        # 4. Бинаризация (пороговая обработка)
        # Если нужно бинаризовать изображение, раскомментируйте:
        # threshold = 150
        # img = img.point(lambda x: 255 if x > threshold else 0)
        
        # 5. Удаление шума (легкое размытие)
        # img = img.filter(ImageFilter.MedianFilter(size=1))
        
        # 6. Увеличение разрешения (если изображение мелкое)
        # width, height = img.size
        # if width < 1000 or height < 1000:
        #     img = img.resize((width*2, height*2), Image.Resampling.LANCZOS)
        
        # 7. Коррекция яркости
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)  # Немного увеличиваем яркость
        
        return img

    # Предобработка изображения
    img = preprocess_image('1/1_Страница_01_Изображение_0001.png')
    
    # Опционально: сохраняем предобработанное изображение для проверки
    # img.save('preprocessed_image.png')
    
    # Настройки Tesseract для улучшения распознавания
    # custom_config = r'--oem 3 --psm 6'  # oem 3 = авто, psm 6 = блок текста
    
    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(img, 
                                      lang='rus'  # Можно добавить несколько языков
                                    )

    # Print the extracted text
    print("=" * 50)
    print("РАСПОЗНАННЫЙ ТЕКСТ:")
    print("=" * 50)
    print(text)
    print("=" * 50)

except ImportError as e:
    print(f"Error importing a module: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    print("Ensure the Tesseract executable is installed and in your system's PATH, or set pytesseract.pytesseract.tesseract_cmd manually.")
except Exception as e:
    print(f"An error occurred: {e}")