import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import joblib
from tqdm import tqdm

# --- Configuration ---
IMAGE_DIR = 'images'
TRAIN_CSV_PATH = 'train.csv'
MODEL_SAVE_DIR = 'saved_model'
MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'plant_health_model.joblib')
CLASS_NAMES_PATH = os.path.join(MODEL_SAVE_DIR, 'class_names.joblib')
IMG_SIZE = (224, 224)

# --- Feature Extraction ---
def extract_features(image_path, img_size):
    """
    Extracts handcrafted features (Color and Texture) from an image.
    """
    try:
        # Load and resize image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None
        img_resized = cv2.resize(img, img_size)

        # 1. Color Features (in HSV color space)
        hsv_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        mean_h, std_h = np.mean(h), np.std(h)
        mean_s, std_s = np.mean(s), np.std(s)
        mean_v, std_v = np.mean(v), np.std(v)

        color_features = [mean_h, std_h, mean_s, std_s, mean_v, std_v]

        # 2. Texture Features (GLCM)
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # GLCM needs integer values
        gray_img_int = gray_img.astype(np.uint8)
        
        glcm = graycomatrix(gray_img_int, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        texture_features = [contrast, dissimilarity, homogeneity, energy, correlation]

        # Combine all features into a single vector
        return np.hstack([color_features, texture_features])

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- Main Training Logic ---
def train_model():
    """
    Loads data, extracts features, trains a model, and saves it.
    """
    print("Loading training data...")
    df = pd.read_csv(TRAIN_CSV_PATH)

    # Convert one-hot encoded labels to a single categorical column
    label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
    df['label'] = df[label_cols].idxmax(axis=1)
    
    # Store class names for later use in the app
    class_names = df['label'].unique().tolist()
    class_names.sort() # Sort for consistency
    print(f"Found classes: {class_names}")

    # Create directories if they don't exist
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
        
    print("Extracting features from images... This might take a while.")
    features_list = []
    labels_list = []

    # Use tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Images"):
        image_id = row['image_id']
        label = row['label']
        image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")

        if os.path.exists(image_path):
            features = extract_features(image_path, IMG_SIZE)
            if features is not None:
                features_list.append(features)
                labels_list.append(label)
        else:
            print(f"Warning: Image file not found at {image_path}")
            
    if not features_list:
        print("Error: No features were extracted. Check your image directory and paths.")
        return

    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\nFeature extraction complete. Shape of feature matrix: {X.shape}")

    # Splitting data into training and validation sets
    print("Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained. Validation Accuracy: {accuracy:.4f}")

    # Save the trained model and class names
    print(f"Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(class_names, CLASS_NAMES_PATH)
    print("Model and class names saved successfully.")

if __name__ == '__main__':
    train_model()