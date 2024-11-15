import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained MNIST model
model = tf.keras.models.load_model('C:/Users/daksh/OneDrive/Desktop/ML Projects/MNIST Handwritten Digits Classification/Trained Model/trained_mnist_handwritten_digit_classification_model.h5')

# Streamlit app
st.title("MNIST Handwritten Digit Classification")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image to 28x28 pixels and normalize the data
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the model

    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    
    st.write(f"The model predicts the digit as: {predicted_digit}")
