try:
    from PIL import Image
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ordum\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


    # Open an image file
    img = Image.open('1/1_Страница_01_Изображение_0001.png')

    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(img, lang='rus')


    # Print the extracted text
    print(text)

except ImportError as e:
    print(f"Error importing a module: {e}")
except FileNotFoundError:
    print("Ensure the Tesseract executable is installed and in your system's PATH, or set pytesseract.pytesseract.tesseract_cmd manually.")
except Exception as e:
    print(f"An error occurred: {e}")