
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, models
from PIL import Image
import io

st.title("Real vs Fake Image Classifier")
st.write("Upload an image and the model will predict if it's real or fake.")

# --- Model Loading (Re-defining and loading weights as previously done) ---

# Define the MobileNetV2 base model
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Rebuild the Sequential model architecture
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# Load the weights into the defined model
try:
    model.load_weights("/content/RealVsFake_MobileNetV2.h5")
    st.success("Model weights loaded successfully!")
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop() # Stop the app if model loading fails

# --- Image Upload and Prediction ---

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_resized = img.resize((224, 224))
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0) # Add batch dimension
    x = preprocess_input(x)

    # Make prediction
    prediction = model.predict(x)
    probability = prediction[0][0]

    # Interpret result
    if probability > 0.5:
        st.error(f"Prediction: Fake (Confidence: {probability:.2f})")
    else:
        st.success(f"Prediction: Real (Confidence: {1 - probability:.2f})")
