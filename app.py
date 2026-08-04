import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import pandas as pd

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM STYLE
# ---------------------------------------------------
st.markdown(
    """
    <style>
    /* Reduce overall font size */
    html, body, [class*="css"]  {
        font-size: 14px !important;
    }

    /* Smaller titles */
    h1, h2, h3 {
        font-size: 18px !important;
    }

    /* Sidebar compact */
    .css-1d391kg, .css-1v3fvcr {
        font-size: 13px !important;
    }

    /* Reduce padding/margins */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

classifier = joblib.load("models/skin_cancer_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
cnn = tf.keras.models.load_model(
    "models/mobilenet_feature_extractor.keras"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🩺 Skin Cancer Classifier")

st.sidebar.markdown("""
## Project Information

**Developer:**  
Hadia Sohail

**Dataset:**  
HAM10000

**Feature Extractor:**  
MobileNetV2

**Classifier:**  
Support Vector Machine (SVM)

**Classes:**  
7 Skin Lesion Types
""")

st.sidebar.markdown("---")

st.sidebar.write("""
### Pipeline

Image

⬇

MobileNetV2

⬇

Feature Extraction

⬇

StandardScaler

⬇

Support Vector Machine (SVM)

⬇

Prediction
""")

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title("🩺 Skin Cancer Classification System")

st.markdown("""
Upload a dermoscopic skin lesion image.

This application uses **MobileNetV2** for deep feature extraction and a **Support Vector Machine (SVM)** for skin lesion classification.
""")

# ---------------------------------------------------
# DISEASE INFORMATION
# ---------------------------------------------------

disease_names = {
    "akiec": "Actinic Keratoses and Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus (Benign Mole)",
    "vasc": "Vascular Lesion"
}

description = {
    "akiec": "Precancerous skin lesion that may develop into squamous cell carcinoma.",
    "bcc": "Most common skin cancer with slow growth and low metastatic risk.",
    "bkl": "A non-cancerous skin growth.",
    "df": "A benign fibrous skin nodule.",
    "mel": "A dangerous type of skin cancer requiring immediate medical attention.",
    "nv": "A common benign mole with very low cancer risk.",
    "vasc": "Benign lesion caused by blood vessels."
}

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Skin Lesion Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        processed = image.resize((224, 224))
        st.image(processed, use_container_width=True)

    img_array = np.array(processed)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):

        features = cnn.predict(img_array, verbose=0)
        features = features.reshape(1, -1)

        features = scaler.transform(features)

        prediction = classifier.predict(features)
        probabilities = classifier.predict_proba(features)[0]

        predicted_label = label_encoder.inverse_transform(prediction)[0]

    confidence = np.max(probabilities) * 100

    dangerous = ["mel", "bcc", "akiec"]

    st.markdown("---")

    if predicted_label in dangerous:
        st.error(f"### Prediction: {disease_names[predicted_label]}")
    else:
        st.success(f"### Prediction: {disease_names[predicted_label]}")

    st.info(f"### Confidence: {confidence:.2f}%")

    st.subheader("Disease Information")

    st.write(description[predicted_label])

    # ---------------------------------------------------
    # PROBABILITY CHART
    # ---------------------------------------------------

    st.subheader("Prediction Probability")

    probability_df = pd.DataFrame({
        "Disease": [disease_names[label] for label in label_encoder.classes_],
        "Probability (%)": probabilities * 100
    })

    probability_df = probability_df.set_index("Disease")

    st.bar_chart(probability_df)

# ---------------------------------------------------
# ABOUT PROJECT
# ---------------------------------------------------

with st.expander("📖 About this Project"):

    st.write("""
This application classifies dermoscopic skin lesion images into **seven skin lesion classes**.

### Workflow

1. Upload Image
2. Resize to 224×224
3. MobileNetV2 Feature Extraction
4. StandardScaler
5. Support Vector Machine (SVM)
6. Prediction

**Dataset:** HAM10000

**Developer:** Hadia Sohail
""")

# ---------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------

st.markdown("---")

st.warning("""
⚠ **Medical Disclaimer**

This application is intended for educational and research purposes only.

It is **NOT** a substitute for professional medical diagnosis.

Always consult a qualified dermatologist before making healthcare decisions.
""")