import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = r"E:\GSOC2025\best_model_epoch_100.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names mapping
class_names = {
    "artist": {
        0: "Albrecht Durer", 1: "Boris Kustodiev", 2: "Camille Pissarro", 3: "Childe Hassam",
        4: "Claude Monet", 5: "Edgar Degas", 6: "Eugene Boudin", 7: "Gustave Dore",
        8: "Ilya Repin", 9: "Ivan Aivazovsky", 10: "Ivan Shishkin", 11: "John Singer Sargent",
        12: "Marc Chagall", 13: "Martiros Saryan", 14: "Nicholas Roerich", 15: "Pablo Picasso",
        16: "Paul Cezanne", 17: "Pierre Auguste Renoir", 18: "Pyotr Konchalovsky", 19: "Raphael Kirchner",
        20: "Rembrandt", 21: "Salvador Dali", 22: "Vincent van Gogh"
    },
    "genre": {
        0: "Abstract Painting", 1: "Cityscape", 2: "Genre Painting", 3: "Illustration",
        4: "Landscape", 5: "Nude Painting", 6: "Portrait", 7: "Religious Painting",
        8: "Sketch and Study", 9: "Still Life"
    },
    "style": {
        0: "Abstract Expressionism", 1: "Action Painting", 2: "Analytical Cubism", 3: "Art Nouveau",
        4: "Baroque", 5: "Color Field Painting", 6: "Contemporary Realism", 7: "Cubism",
        8: "Early Renaissance", 9: "Expressionism", 10: "Fauvism", 11: "High Renaissance",
        12: "Impressionism", 13: "Mannerism Late Renaissance", 14: "Minimalism",
        15: "Naive Art Primitivism", 16: "New Realism", 17: "Northern Renaissance",
        18: "Pointillism", 19: "Pop Art", 20: "Post Impressionism", 21: "Realism",
        22: "Rococo", 23: "Romanticism", 24: "Symbolism", 25: "Synthetic Cubism", 26: "Ukiyo-e"
    }
}

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension -> (1, 128, 128, 3)
    
    # If the model expects a sequence (10 frames), duplicate the image 10 times
    if model.input_shape[1] == 10:
        image = np.repeat(image[np.newaxis, :, :, :, :], 10, axis=1)  # Shape: (1, 10, 128, 128, 3)

    return image

# Streamlit UI
st.title("🎨 AI Art Classifier")
st.write("Upload an art image to classify the **Artist, Style, and Genre**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_image = preprocess_image(image)

    # Get predictions
    predictions = model.predict(input_image)

    # Extract predictions
    artist_pred = np.argmax(predictions[0])
    genre_pred = np.argmax(predictions[1])
    style_pred = np.argmax(predictions[2])

    # Display results
    st.subheader(" <------- PREDICTIONS --------->")
    st.write(f"🎨 **Artist:** {class_names['artist'][artist_pred]}")
    st.write(f"🖌 **Style:** {class_names['style'][style_pred]}")
    st.write(f"🎭 **Genre:** {class_names['genre'][genre_pred]}")
