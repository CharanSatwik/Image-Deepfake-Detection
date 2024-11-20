import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model("deepfake-detector-model.keras")

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app
st.title("Image Deepfake Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_labels = np.round(prediction)

    st.write(f"Predicted Label: {'Fake' if prediction[0][0] > 0.61 else 'Real'}")
    


    # Display the image and prediction using matplotlib within Streamlit
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f"Predicted Label: {'Fake' if prediction[0][0] > 0.61 else 'Real'}")
    st.pyplot(fig)
