import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib

# Load your trained models
digit_model = tf.keras.models.load_model("C:/Users/daksh/OneDrive/Desktop/ML Projects/MNIST Handwritten Digits Classification/Trained Model/trained_mnist_handwritten_digit_classification_model.h5")
operator_model = joblib.load("C:/Users/daksh/OneDrive/Desktop/ML Projects/MNIST Handwritten Digits Classification/operator_detector.joblib")

# Streamlit app
st.title("Handwritten Math Equation Evaluator")

# File uploaders for digit and operator images
st.write("Upload two images of digits and one image of a mathematical operator.")
uploaded_digit1 = st.file_uploader("Upload first digit image", type=["png", "jpg", "jpeg"])
uploaded_operator = st.file_uploader("Upload operator image", type=["png", "jpg", "jpeg"])
uploaded_digit2 = st.file_uploader("Upload second digit image", type=["png", "jpg", "jpeg"])

# Function to preprocess image for digit model
def preprocess_digit_image(image):
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Normalize
    return image_array.reshape(1, 28, 28, 1)  # Reshape for the model

# Function to preprocess image for operator model
def preprocess_operator_image(image):
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image).flatten()  # Flatten to 1D for the classifier
    return image_array.reshape(1, -1)

# Check if all files are uploaded
if uploaded_digit1 and uploaded_operator and uploaded_digit2:
    # Display uploaded images
    image1 = Image.open(uploaded_digit1).convert('L')
    st.image(image1, caption="First Digit Image", use_column_width=True)
    image_op = Image.open(uploaded_operator).convert('L')
    st.image(image_op, caption="Operator Image", use_column_width=True)
    image2 = Image.open(uploaded_digit2).convert('L')
    st.image(image2, caption="Second Digit Image", use_column_width=True)
    
    # Predict the digits
    image1_array = preprocess_digit_image(image1)
    image2_array = preprocess_digit_image(image2)
    pred_digit1 = np.argmax(digit_model.predict(image1_array))
    pred_digit2 = np.argmax(digit_model.predict(image2_array))
    
    # Predict the operator
    operator_array = preprocess_operator_image(image_op)
    pred_operator = operator_model.predict(operator_array)[0]
    operator_dict = {0: "+", 1: "-", 2: "*"}  # Adjust to match your operator encoding
    operator_symbol = operator_dict.get(pred_operator, "Unknown")
    
    # Display the equation and calculate the result
    if operator_symbol != "Unknown":
        equation = f"{pred_digit1} {operator_symbol} {pred_digit2}"
        try:
            result = eval(equation)  # Safely evaluate the expression
            st.write(f"Equation: {equation}")
            st.write(f"Result: {result}")
        except Exception as e:
            st.write("Error in evaluating the expression.")
    else:
        st.write("Unable to recognize the operator.")
