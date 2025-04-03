import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

CLASS_MAPPING = {
    0: "actinic keratosis",
    1: "basal cell carcinoma",
    2: "dermatofibroma",
    3: "melanoma",
    4: "nevus",
    5: "pigmented benign keratosis",
    6: "seborrheic keratosis",
    7: "squamous cell carcinoma",
    8: "vascular lesion"
}

model = tf.keras.models.load_model("/Users/nandanpatel/Desktop/Skin_Cancer/model.h5")


def preprocess_image(img):
    img = img.resize((150, 150))  # Adjust size based on model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


st.title("Skin Cancer Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_resized = img.resize((150, 150))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    img_array = preprocess_image(img)
    y_pred_probs = model.predict(img_array)
    y_pred = np.argmax(y_pred_probs, axis=1)[0]
    class_name = CLASS_MAPPING.get(y_pred, "Unknown")

    st.write(f"Predicted Class: {class_name}")