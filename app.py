import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Load your model (with caching to improve performance)
# @st.cache_resource()
model = tf.keras.models.load_model("disease_prediction_model.h5")

# loading the class names
import json

with open('class_indices.json', 'r') as f:  # Open the file in read mode
    class_indices = json.load(f)



# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(240, 240)):
    # Check if image_path is already a PIL Image object
    if isinstance(image_path, Image.Image):
        img = image_path
    else:
        # Load the image if it's a path
        img = Image.open(image_path)
    # # Load the image
    # img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('Mango Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((240, 240))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
