import cv2
import numpy as np
from paddleocr import PaddleOCR
from deep_translator import GoogleTranslator
from PIL import ImageFont, ImageDraw, Image

# 1️⃣ 初始化 OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
)

# 2️⃣ 执行识别
result = ocr.predict("img_1.png")

print(result)
# 3️⃣ 取出主要结果对象
res = result[0]['res'] if 'res' in result[0] else result[0]

# 4️⃣ 获取图片（OpenCV 图像）
img_array = res['doc_preprocessor_res']['output_img']
img = img_array.copy()

# 5️⃣ 初始化中文字体（OpenCV 默认不支持中文）
# 所以使用 cv2.putTextNatively 的替代方案
# 用 PIL 绘制中文，再转换回 OpenCV

# font_path = "C:/Windows/Fonts/simhei.ttf"
font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'

font = ImageFont.truetype(font_path, 18)


def draw_text_chinese(img_cv2, text, position, font, color=(0, 0, 0)):
    """在 OpenCV 图像上绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# 6️⃣ 翻译器
translator = GoogleTranslator(source="auto", target="zh-CN")

# 7️⃣ 提取识别文本与坐标
texts = res['rec_texts']
polys = res['rec_polys']

for text, box in zip(texts, polys):
    print()
    try:
        translated = translator.translate(text)
    except Exception as e:
        print("翻译失败：", e)
        translated = text
        continue

    if not translated:
        continue

    print(f"{text} -> {translated}")

    # 坐标点
    pts = np.array(box, dtype=np.int32)

    # 覆盖原文字区域（白底）
    cv2.fillPoly(img, [pts], color=(255, 255, 255))

    # 文字绘制位置
    x_min = np.min(pts[:, 0])
    y_min = np.min(pts[:, 1])

    # 绘制译文（中文）
    img = draw_text_chinese(img, translated, (x_min, y_min), font, (0, 0, 0))

# 8️⃣ 保存输出图像
cv2.imwrite("translated.png", img)
print("✅ 翻译完成，输出文件：translated.png")
