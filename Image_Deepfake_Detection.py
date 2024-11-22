import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained deepfake detection model
model = tf.keras.models.load_model("deepfake-detector-model.keras")

def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_deepfake(image_path):
    try:
        image = Image.open(image_path)
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_label = prediction[0][0]

        if predicted_label >= 0.61:
            return "Fake"
        else:
            return "Real"
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error"

def main():
    st.title("Deepfake Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        prediction_result = predict_deepfake(uploaded_file)

        if prediction_result == "Error":
            st.error("An error occurred while processing the image.")
        else:
            st.write(f"Predicted Label: {prediction_result}")

if __name__ == "__main__":
    main()
