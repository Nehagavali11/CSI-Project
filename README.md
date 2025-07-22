# Plant Leaf Health Classifier 🌿

This project delivers a **Streamlit** web application for classifying plant leaf health. It uses traditional machine learning with handcrafted image features and integrates an **AI assistant** powered by Google's Gemini model.

---
## Live App : - https://plant-species-classification-csi.streamlit.app

---

## Technologies Used 💻

* **Python:** Core programming language.
* **Streamlit:** For building the interactive web application.
* **OpenCV & Scikit-image:** For image processing and feature extraction.
* **Pandas & NumPy:** Data handling.
* **Scikit-learn:** Machine learning models.
* **Joblib:** Model saving/loading.
* **Plotly Express:** Data visualization.
* **Google Generative AI:** Powers the AI assistant (`gemini-1.5-flash`).

---

## Methodology ⚙️

### 1. Feature Extraction

* **Handcrafted features** are extracted from leaf images:
    * **Color Features (HSV):** Mean and standard deviation of color channels.
    * **Texture Features (GLCM):** Measures like Contrast, Homogeneity, and Correlation to identify surface patterns.

### 2. Machine Learning Model

* **Algorithm:** A **Random Forest Classifier** is used to predict plant health status (healthy, multiple diseases, rust, scab).
* **Training & Saving:** The model is trained on extracted features and saved for use in the app.

### 3. Application Features

* **Exploratory Data Analysis (EDA):** Visualize dataset characteristics.
* **Image Classification:** Upload a leaf image to get an immediate health prediction with confidence.
* **AI Assistant:** Ask general questions about plant care and diseases.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
---

![Plant Health Classifier · Streamlit_page-0001](https://github.com/user-attachments/assets/8776fbef-6241-44a3-9940-1f4f52631533)

![Plant Health Classifier · Streamlit_page-0002](https://github.com/user-attachments/assets/5f02ee15-0bff-43b7-86b1-06e0f6901032)

![Plant Health Classifier · Streamlit1_page-0003](https://github.com/user-attachments/assets/3b682b1f-0d3c-4723-847d-adc492d7709a)

![Plant Health Classifier · Streamlit1_page-0004](https://github.com/user-attachments/assets/e83a671d-035e-432c-9879-2fdc56a36af9)

![Plant Health Classifier · Streamlit2_page-0005](https://github.com/user-attachments/assets/259f7be7-7550-4256-a813-2bfb5299aa1c)

![Plant Health Classifier · Streamlit2_page-0006](https://github.com/user-attachments/assets/5ea870f7-c5d9-4005-9628-209efc05c06d)




