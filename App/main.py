import os
import  json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/Animal Classification.h5"

#load the pre-trained model
model = tf.keras.models.load_model(model_path)

#loading the class names
class_indices = json.load(open(f"{working_dir}/classes.json"))

# === Image Preprocessing Function ===
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, resample=Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array

# === Prediction Function ===
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = class_indices[str(predicted_index)]
    confidence = predictions[0][predicted_index]
    return predicted_label, confidence

# === Streamlit UI ===
st.set_page_config(page_title="Animal Classifier", page_icon="🦁", layout="centered")

st.markdown("<h1 style='text-align: center;'>🦁 Animal Classifier 🐘</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an animal image and let AI identify the species!</p>", unsafe_allow_html=True)
st.markdown("---")

# === Sidebar ===
st.sidebar.title("📘 Instructions")
st.sidebar.info("1. Upload a JPG or PNG image.\n\n2. Click **Classify**.\n\n3. View predicted animal and confidence score.")

# === Main Area ===
upload_image = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png"])

if upload_image is not None:
    image = Image.open(upload_image).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((200, 200), resample=Image.Resampling.LANCZOS)
        st.image(resized_image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("Classify"):
            predicted_label, confidence = predict_image_class(model, upload_image, class_indices)
            st.success(f"Prediction: **{predicted_label}**")
            st.info(f"Confidence: **{confidence * 100:.2f}%**")

    st.markdown("<div style='text-align: center;'>👈 Upload an image to start classification.</div>", unsafe_allow_html=True)