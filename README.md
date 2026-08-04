# 🩺 Skin Cancer Classification using MobileNetV2 and SVM

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-SVM-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Local-red)
![License](https://img.shields.io/badge/License-MIT-green)

<img width="947" height="437" alt="image" src="https://github.com/user-attachments/assets/487c2028-fe5b-4a4f-b4f9-9076b766db69" />

> A hybrid Deep Learning and Machine Learning system for multiclass skin lesion classification using the HAM10000 dataset. The project combines **MobileNetV2** for deep feature extraction with a **Support Vector Machine (SVM)** classifier and provides a **locally executed Streamlit interface** for image-based prediction and disease information display.

---

# 📌 Project Overview

Skin cancer is among the most common forms of cancer worldwide, and early diagnosis significantly improves treatment outcomes. Manual examination of dermoscopic images requires specialized expertise and may be time-consuming.

This project presents a hybrid Artificial Intelligence approach for automated skin lesion classification. Instead of using an end-to-end CNN classifier, the system utilizes **MobileNetV2** as a feature extractor to generate deep image representations. These extracted features are then classified using a **Support Vector Machine (SVM)** to identify different categories of skin lesions.

To make the system user-friendly, a **Streamlit web interface** was developed and tested locally using **Visual Studio Code**, allowing users to upload dermoscopic images, obtain predictions, and view brief disease-related information.
[streamlit-app-2026-08-04-21-59-27.webm](https://github.com/user-attachments/assets/f3ac2219-b258-4b10-ab51-8efb27dbc5f4)


---

# 📍 Current Status

✅ Model Development Completed

✅ MobileNetV2 Feature Extraction

✅ SVM Classifier Training Completed

✅ Streamlit Web Interface Developed

✅ Local Testing Completed using VS Code

✅ Google Colab Notebook Included


---

# 🎯 Objectives

- Develop an automated skin lesion classification system.
- Extract robust image features using MobileNetV2.
- Classify lesions using Support Vector Machine (SVM).
- Build an easy-to-use Streamlit interface.
- Assist in preliminary skin lesion assessment.
- Demonstrate the practical application of AI in healthcare.

---

# ✨ Features

- ✅ Hybrid Deep Learning + Machine Learning Model
- ✅ MobileNetV2 Feature Extraction
- ✅ SVM Multi-class Classification
- ✅ HAM10000 Dataset
- ✅ Local Streamlit Web Interface
- ✅ Image Upload Functionality
- ✅ Real-time Prediction
- ✅ Disease Information Display
- ✅ Google Colab Training Notebook

---

# 🏗 Project Workflow

```text
Input Skin Lesion Image
          │
          ▼
Image Preprocessing
          │
          ▼
MobileNetV2 Feature Extraction
          │
          ▼
Deep Feature Vector
          │
          ▼
Support Vector Machine (SVM)
          │
          ▼
Predicted Skin Lesion Class
          │
          ▼
Disease Information Display
```

---

# 📂 Dataset

## Dataset Used

**HAM10000 (Human Against Machine with 10000 Training Images)**

The HAM10000 dataset contains over **10,000 dermoscopic images** collected from multiple clinical sources and covers seven different categories of pigmented skin lesions.

## Classes

| Code | Disease |
|------|------------------------------|
| akiec | Actinic Keratoses |
| bcc | Basal Cell Carcinoma |
| bkl | Benign Keratosis |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic Nevus |
| vasc | Vascular Lesions |

Dataset:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

---

# 🧠 Model Architecture

## Feature Extractor

- MobileNetV2
- Image Size: 224 × 224
- Transfer Learning
- ImageNet Pre-trained Weights

## Classifier

- Support Vector Machine (SVM)
- Multi-class Classification
- Trained using extracted deep features

---

# ⚙ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Scikit-learn
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Pillow
- Joblib

---

# 📒 Google Colab Notebook

The complete training pipeline, preprocessing, feature extraction, model development, and evaluation are available in the notebook.

Notebook Location

```
notebook/SkinCancerClassification.ipynb
```

If a public Colab notebook is available, add the link here.

---

# 💻 Local Demonstration

The application was developed and tested locally using:

- Visual Studio Code
- Python
- Streamlit

Run the application locally using:

```bash
streamlit run app.py
```

Users can:

- Upload a dermoscopic skin lesion image.
- Receive the predicted lesion category.
- View disease-related information.
- Test the trained MobileNetV2 + SVM model.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Skin-Cancer-Classification.git
```

Go to the project directory

```bash
cd Skin-Cancer-Classification
```

Install required libraries

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# 💻 Usage

1. Launch the Streamlit application.
2. Upload a dermoscopic skin lesion image.
3. The image is preprocessed.
4. MobileNetV2 extracts deep image features.
5. The trained SVM predicts the lesion class.
6. The application displays:
   - Predicted Class
   - Disease Information

---

# 📊 Model Performance

> # 📊 Model Performance Comparison

The performance of multiple machine learning classifiers was evaluated using features extracted from MobileNetV2. The **Support Vector Machine (SVM)** achieved the highest overall performance and was therefore selected as the final classifier for the skin cancer classification system.

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|:--------:|:---------:|:------:|:--------:|
| 🥇 Support Vector Machine (SVM) | **78.08%** | **75.06%** | **78.08%** | **74.96%** |
| K-Nearest Neighbors (KNN) | 72.89% | 69.93% | 72.89% | 70.08% |
| Logistic Regression | 71.84% | 72.22% | 71.84% | 71.97% |
| Decision Tree | 62.36% | 62.58% | 62.36% | 62.46% |
| Naive Bayes | 43.63% | 70.83% | 43.63% | 51.36% |

### Key Findings

- **Support Vector Machine (SVM)** achieved the highest overall performance with an **accuracy of 78.08%**, making it the best-performing classifier.
- **K-Nearest Neighbors (KNN)** ranked second with an accuracy of **72.89%**.
- **Logistic Regression** showed competitive performance with **71.84% accuracy**.
- **Decision Tree** and **Naive Bayes** performed comparatively lower, indicating that they were less suitable for this feature representation.
- Based on these results, the **MobileNetV2 + SVM hybrid model** was selected as the final model the Streamlit application.

---

# 🖼 Application Screenshots

## Home Page

<img width="923" height="403" alt="image" src="https://github.com/user-attachments/assets/4eaaefdf-de93-4587-bb0e-ae0ec2d04d9a" />


---

## Image Upload

<img width="600" height="450" alt="sample" src="https://github.com/user-attachments/assets/00127be4-f09d-454f-b99b-a00f77e81d67" />
---

## Prediction Result

<img width="663" height="165" alt="image" src="https://github.com/user-attachments/assets/16cd0183-56a7-425b-8924-ca2fc14b8f7b" />
<img width="669" height="305" alt="image" src="https://github.com/user-attachments/assets/e585e326-22ef-4578-aed1-df93beba3307" />


# 🔬 Future Improvements

- Deploy the application using Streamlit Community Cloud.
- Improve classification accuracy using EfficientNet.
- Implement Grad-CAM for explainable predictions.
- Develop a mobile application.
- Expand the dataset with additional dermoscopic images.
- Integrate patient history for enhanced decision support.

---

# 📚 References

1. HAM10000 Dataset

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

2. MobileNetV2 Paper

https://arxiv.org/abs/1801.04381

3. TensorFlow Documentation

https://www.tensorflow.org/

4. Streamlit Documentation

https://streamlit.io/

---

# 👨‍💻 Author

**Hadia Sohail**

Biomedical Engineering Student

University of Engineering and Technology (UET), Lahore

GitHub

https://github.com/hadiasohail793

LinkedIn

www.linkedin.com/in/hadia-sohail


# 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project useful, please consider giving it a star!
---

⭐ If you found this project useful, consider giving the repository a star!
