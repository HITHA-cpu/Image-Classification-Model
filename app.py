# save this as app.py
import streamlit as st
import cv2
import numpy as np
import json
import joblib
import pywt
from PIL import Image

# ====== CONFIG ======
MODEL_PATH = "saved_model.pkl"
CLASS_DICT_PATH = "class_dictionary.json"

# ====== FUNCTIONS ======
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray) / 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    return np.uint8(imArray_H)

def preprocess_image(img):
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((
        scalled_raw_img.reshape(32*32*3, 1),
        scalled_img_har.reshape(32*32, 1)
    ))
    return combined_img.reshape(1, -1).astype(float)

# ====== LOAD MODEL & CLASS DICTIONARY ======
model = joblib.load(MODEL_PATH)
with open(CLASS_DICT_PATH, "r") as f:
    class_dict = json.load(f)
reverse_class_dict = {v: k for k, v in class_dict.items()}

# ====== STREAMLIT UI ======
st.title("Celebrity Face Classifier 🎯")
st.write("Upload a cropped celebrity image and I’ll tell you who it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    features = preprocess_image(img_array)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features).max()

    st.subheader(f"Prediction: {reverse_class_dict[prediction]}")
    st.write(f"Confidence: {probability:.2f}")
