from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Инициализация
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Загрузка изображения
image = Image.open("1/1_1.png").convert("RGB")

# Обработка и генерация
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

# Для простого распознавания текста
task_prompt = "<s>"
decoder_input_ids = processor.tokenizer(
    task_prompt, 
    add_special_tokens=False, 
    return_tensors="pt"
).input_ids.to(device)

outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=512,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

# Вывод результата
sequence = processor.batch_decode(outputs)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = sequence.replace("<s>", "").replace("</s>", "").strip()

print("Распознанный текст:", sequence)