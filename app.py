import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2

# Load trained SVM model
svm_model = joblib.load("skin_cancer_model.pkl")

st.title("Skin Cancer Classifier (SVM Only)")

uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (resize, flatten)
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (128, 128))  # match training size
    features = img_resized.flatten().reshape(1, -1)

    prediction = svm_model.predict(features)
    st.write("Prediction:", prediction[0])
