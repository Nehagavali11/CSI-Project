# Plant Leaf Health Classifier 🌿

This project delivers a **Streamlit** web application for classifying plant leaf health. It uses traditional machine learning with handcrafted image features and integrates an **AI assistant** powered by Google's Gemini model.

---
## Live App : - 

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
