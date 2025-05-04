%%writefile app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# إعدادات
IMAGE_SIZE = 64

# تحميل النموذج
@st.cache_resource
def load_asl_model():
    model = load_model("asl_model.h5")
    return model

model = load_asl_model()

# القاموس العكسي للحروف
label_map_ar = [
    'ع', 'ال', 'أ', 'ب', 'د', 'ض', 'ف', 'غ', 'ح', 'ه',
    'ج', 'ك', 'خ', 'لا', 'ل', 'م', 'ن', 'ق', 'ر', 'ص',
    'س', 'ش', 'ط', 'ت', 'ة', 'ذ', 'ث', 'و', 'ي', 'ظ', 'ز'
]

reverse_label_map = {i: label for i, label in enumerate(sorted(label_map_ar))}

# دالة لكتابة نص عربي على صورة OpenCV باستخدام Pillow
def draw_arabic_text(img, text, position, font_path="Amiri-Regular.ttf", font_size=36, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# واجهة Streamlit
st.title("🔤 مترجم لغة الإشارة - كاميرا الويب أو صورة")
st.markdown("إبداع بلا صوت*")

# وظيفة التنبؤ بالحرف من صورة
def predict_from_image(image: Image.Image):
    gray = image.convert('L')
    resized = gray.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(resized).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
    prediction = model.predict(img_array)
    pred_label = reverse_label_map[np.argmax(prediction)]
    return pred_label

# اختيار الوضع: كاميرا أو صورة
mode = st.radio("اختر المصدر:", ["📷 كاميرا", "🖼️ صورة"])

if mode == "📷 كاميرا":
    class SignLanguageTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            center_crop = gray[h//2-100:h//2+100, w//2-100:w//2+100]

            try:
                resized = cv2.resize(center_crop, (IMAGE_SIZE, IMAGE_SIZE)).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1) / 255.0
                prediction = model.predict(resized)
                pred_label = reverse_label_map[np.argmax(prediction)]

                reshaped_text = arabic_reshaper.reshape(f"الإخراج: {pred_label}")
                bidi_text = get_display(reshaped_text)
                img = draw_arabic_text(img, bidi_text, (10, 40), font_size=36)

                # إضافة الشعار
                logo_text = arabic_reshaper.reshape("إبداع بلا صوت")
                logo_bidi = get_display(logo_text)
                img = draw_arabic_text(img, logo_bidi, (img.shape[1]-300, 10), font_size=28, color=(255, 255, 255))

                # مربع اليد
                cv2.rectangle(img, (w//2-100, h//2-100), (w//2+100, h//2+100), (255, 0, 0), 2)
            except Exception as e:
                print("خطأ:", e)

            return img

    webrtc_streamer(key="sign-detect", video_transformer_factory=SignLanguageTransformer)

else:
    uploaded_file = st.file_uploader("ارفع صورة لليد بلغة الإشارة", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="الصورة المرفوعة", use_column_width=True)

        pred_label = predict_from_image(img)
        reshaped_text = arabic_reshaper.reshape(f"الإخراج: {pred_label}")
        bidi_text = get_display(reshaped_text)
        st.markdown(f"### ✅ النتيجة: **{bidi_text}**")
