import streamlit as st
import pandas as pd
import numpy as np
import cv2
import joblib
import os
import plotly.express as px
import google.generativeai as genai
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Plant Health Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Google Gemini API Configuration ---
# WARNING: Do not expose your API key publicly. 
# Use Streamlit secrets for deployment.
# For local testing, it's okay, but be careful.
API_KEY = "AIzaSyCCG8qJdwQHPVnaBXkwU5xl8B_xLppBVrI" 
try:
    genai.configure(api_key=API_KEY)
    # Changed 'gemini-pro' to 'gemini-1.5-flash' for broader availability
    gemini_model = genai.GenerativeModel('gemini-1.5-flash') 
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. The AI Assistant will not be available.")
    gemini_model = None


# --- File Paths ---
MODEL_PATH = os.path.join('saved_model', 'plant_health_model.joblib')
CLASS_NAMES_PATH = os.path.join('saved_model', 'class_names.joblib')
TRAIN_CSV_PATH = 'train.csv'
IMAGE_DIR = 'images'

# --- Feature Extraction (Must match train_model.py) ---
# Re-using the function from the training script for consistency
def extract_features(image_array, img_size=(224, 224)):
    """
    Extracts handcrafted features from an image array.
    """
    # Resize image
    img_resized = cv2.resize(image_array, img_size)
    
    # 1. Color Features (HSV)
    hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    mean_h, std_h = np.mean(h), np.std(h)
    mean_s, std_s = np.mean(s), np.std(s)
    mean_v, std_v = np.mean(v), np.std(v)
    color_features = [mean_h, std_h, mean_s, std_s, mean_v, std_v]

    # 2. Texture Features (GLCM)
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_img_int = gray_img.astype(np.uint8)
    
    from skimage.feature import graycomatrix, graycoprops
    glcm = graycomatrix(gray_img_int, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    texture_features = [contrast, dissimilarity, homogeneity, energy, correlation]

    return np.hstack([color_features, texture_features]).reshape(1, -1)

# --- Caching Functions for Performance ---
@st.cache_data
def load_data(csv_path):
    """Loads the training data and processes labels."""
    df = pd.read_csv(csv_path)
    label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
    # Corrected: Use label_cols list, not string 'label_cols'
    df['label'] = df[label_cols].idxmax(axis=1) 
    return df

@st.cache_resource
def load_model_and_classes():
    """Loads the trained model and class names."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    class_names = joblib.load(CLASS_NAMES_PATH)
    return model, class_names

# --- UI Styling ---
st.markdown("""
<style>
    /* Main title style */
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        background: #F0F2F6;
        color: #333333; /* Darker text for better contrast on light background */
    }
    h1 {
        color: #1E8449;
        text-align: center;
        font-weight: bold;
    }
    h2, h3, h4, h5, h6 {
        color: #2C3E50; /* A darker shade for headings */
    }
    /* Style for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #D5DBDB;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
        color: #566573; /* A medium dark grey for inactive tabs */
    }
    .stTabs [aria-selected="true"] {
        background-color: #D5F5E3;
        border-bottom: 2px solid #1E8449;
        color: #1E8449; /* Green for active tab text */
    }
    /* Custom button style */
    .stButton>button {
        border: 2px solid #1E8449;
        border-radius: 5px;
        color: #1E8449;
        background-color: white;
    }
    .stButton>button:hover {
        border-color: #145A32;
        color: #145A32;
    }
    /* General text color for Streamlit components */
    .css-1d391kg, .css-xq1lnh, .css-1dp5xrc { /* Adjusting general text color for inputs/text areas */
        color: #333333;
    }

    /* Force specific text elements to black */
    /* For st.info, st.success, st.error messages */
    .stAlert p {
        color: black !important; /* Force black text */
    }

    /* For labels of input widgets (slider, text_area, text_input) */
    label {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)


# --- Main Application ---
st.title("🌿 Plant Leaf Health Classifier")
st.markdown("<h4 style='text-align: center; color: #566573;'>Using Traditional Machine Learning & Handcrafted Features</h4>", unsafe_allow_html=True)
st.markdown("---")


# Load model and data
model, class_names = load_model_and_classes()
if model is None or class_names is None:
    st.error("Model not found. Please run `train_model.py` first to generate the model files.")
    st.stop()

df = load_data(TRAIN_CSV_PATH)


# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "🔍 Classify Leaf Image", "⭐ Rate App & AI Assistant"])


# --- Tab 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("Understanding the distribution and characteristics of the plant health dataset.")

    # Show raw data
    with st.expander("View Raw Data"):
        st.dataframe(df)

    col1, col2 = st.columns(2)

    with col1:
        # Class distribution
        st.subheader("Health Status Distribution")
        label_counts = df['label'].value_counts()
        fig = px.bar(label_counts, 
                     x=label_counts.index, 
                     y=label_counts.values,
                     labels={'x': 'Health Status', 'y': 'Number of Images'},
                     color=label_counts.index,
                     color_discrete_map={
                         'healthy': '#2ECC71',
                         'rust': '#E67E22',
                         'scab': '#8E44AD',
                         'multiple_diseases': '#3498DB'
                     })
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Statistical Summary
        st.subheader("Dataset Summary")
        st.write(df.describe())
        # The text "Total Images: ... Number of Categories: ..." will be black due to .stAlert p CSS
        st.info(f"Total Images: **{len(df)}**\n\nNumber of Categories: **{len(class_names)}**")

    st.markdown("---")
    st.subheader("Sample Images from Each Category")
    
    image_cols = st.columns(len(class_names))
    for i, label in enumerate(class_names):
        with image_cols[i]:
            st.markdown(f"<h5 style='text-align: center; color: #34495E;'>{label.replace('_', ' ').title()}</h5>", unsafe_allow_html=True)
            # Find a sample image for this label
            sample_id = df[df['label'] == label].iloc[0]['image_id']
            image_path = os.path.join(IMAGE_DIR, f"{sample_id}.jpg")
            if os.path.exists(image_path):
                # Changed use_column_width to use_container_width
                st.image(image_path, use_container_width=True, caption=f"Example of {label}")
            else:
                st.warning(f"Image for {label} not found.")


# --- Tab 2: Classification ---
with tab2:
    st.header("Classify a New Leaf Image")
    st.markdown("Upload an image of a plant leaf, and the model will predict its health status.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Your Uploaded Image")
            image = Image.open(uploaded_file)
            # Changed use_column_width to use_container_width
            st.image(image, caption="Uploaded Leaf", use_container_width=True)

        # Process and predict
        with st.spinner('Analyzing the leaf...'):
            # Convert PIL image to OpenCV format (BGR)
            image_cv = np.array(image.convert('RGB'))
            image_cv = image_cv[:, :, ::-1].copy() # Convert RGB to BGR

            # Extract features and predict
            features = extract_features(image_cv)
            prediction = model.predict(features)
            probabilities = model.predict_proba(features)

        with col2:
            st.subheader("Prediction Result")
            predicted_class = prediction[0]
            confidence = probabilities.max() * 100

            # The text "Status: ..." will be black due to .stAlert p CSS
            if predicted_class == 'healthy':
                st.success(f"**Status: {predicted_class.title()}** (Confidence: {confidence:.2f}%)")
                st.balloons()
            else:
                st.error(f"**Status: {predicted_class.replace('_', ' ').title()}** (Confidence: {confidence:.2f}%)")

            # Display probabilities
            st.subheader("Prediction Confidence")
            prob_df = pd.DataFrame(probabilities, columns=class_names).T
            prob_df.columns = ["Confidence"]
            prob_df['Confidence'] = prob_df['Confidence'].apply(lambda x: x * 100) # convert to percentage
            
            fig = px.bar(prob_df, x=prob_df.index, y='Confidence',
                         labels={'x': 'Health Status', 'y': 'Confidence (%)'},
                         color=prob_df.index,
                         color_discrete_map={
                             'healthy': '#2ECC71',
                             'rust': '#E67E22',
                             'scab': '#8E44AD',
                             'multiple_diseases': '#3498DB'
                         },
                         text=prob_df['Confidence'].apply(lambda x: f'{x:.2f}%'))
            fig.update_layout(yaxis_title="Confidence (%)", xaxis_title="Health Status")
            st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Feedback and AI Assistant ---
with tab3:
    st.header("Feedback and AI Help")
    
    col1, col2 = st.columns([0.5, 0.5])

    with col1:
        st.subheader("⭐ Rate This Application")
        # Label "How would you rate your experience?" will be black due to label CSS
        rating = st.slider("How would you rate your experience?", 1, 5, 3)
        # Label "Share your feedback or suggestions:" will be black due to label CSS
        feedback_text = st.text_area("Share your feedback or suggestions:")
        if st.button("Submit Feedback"):
            # The text "Thank you for your feedback! We appreciate it." will be black due to .stAlert p CSS
            st.success("Thank you for your feedback! We appreciate it.")
            # In a real app, you would save this feedback to a database or file.
            
    with col2:
        st.subheader("🤖 AI Assistant for Plant Health")
        st.markdown("Ask a general question about plant care or diseases.")
        
        if gemini_model:
            # Label "Your question:" will be black due to label CSS
            user_query = st.text_input("Your question:", placeholder="e.g., How do I prevent rust on my plants?")

            if st.button("Ask AI Assistant"):
                if user_query:
                    with st.spinner("Our AI is thinking..."):
                        # Pre-prompt for context
                        prompt = f"""
                        You are an expert botanist and plant pathologist. 
                        Answer the following user question about plant health in a clear, concise, and helpful way.
                        User Question: "{user_query}"
                        """
                        try:
                            response = gemini_model.generate_content(prompt)
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Could not get a response from the AI. Error: {e}")
                else:
                    st.warning("Please enter a question.")
        else:
            st.info("AI Assistant is currently unavailable. Please check the API key configuration.")

    st.markdown("<p style='text-align: center; color: black; margin-top: 50px;'>App developed by Neha Gavali</p>", unsafe_allow_html=True)