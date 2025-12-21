import easyocr

reader = easyocr.Reader(['ru'])
result = reader.readtext('1/1_1.png')

for detection in result:
    text = detection[1]
    print(text)