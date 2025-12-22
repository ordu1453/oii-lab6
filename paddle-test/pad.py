from paddleocr import PaddleOCR  

ocr = PaddleOCR(
    text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
    use_doc_unwarping=False, # Use use_doc_unwarping to enable/disable document unwarping module
    use_textline_orientation=True, # Use use_textline_orientation to enable/disable textline orientation classification model
    device="gpu:0", # Use device to specify GPU for model inference
)
result = ocr.predict("1/1_10.png")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")

# # Извлекаем все распознанные тексты
# all_texts = []
# for res in result:
#     if hasattr(res, 'rec_texts'):
#         all_texts.extend(res.rec_texts)
#     elif 'rec_texts' in res:
#         all_texts.extend(res['rec_texts'])

# # Выводим все тексты
# for text in all_texts:
#     print(text)

# # Или объединяем в одну строку
# full_text = "\n".join(all_texts)
# print("\nПолный текст:")
# print(full_text)