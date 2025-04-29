import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import h5py

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("keras_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_class_names(path="labels.txt"):
    try:
        with open(path, "r") as f:
            return [line.strip().split(" ", 1)[-1] for line in f.readlines()]
    except FileNotFoundError:
        st.error("Class names file 'labels.txt' not found.")
        return []
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return []

def preprocess_image(image, target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def patch_model_config_if_needed():
    try:
        with h5py.File("keras_model.h5", mode="r+") as f:
            model_config_string = f.attrs.get("model_config").decode() if isinstance(f.attrs.get("model_config"), bytes) else f.attrs.get("model_config")
            if '"groups": 1,' in model_config_string:
                model_config_string = model_config_string.replace('"groups": 1,', '')
                f.attrs.modify('model_config', model_config_string)
                f.flush()
    except Exception as e:
        st.warning(f"Warning patching model config: {e}")

# Patch model config if necessary
patch_model_config_if_needed()

# Streamlit UI
st.title("🔧 Screw Anomaly Detector")
st.markdown("Upload an image of a screw to detect if it is **Good** or **Anomalous**.")

uploaded_file = st.file_uploader("📤 Upload a screw image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    model = load_model()
    if model is None:
        st.stop()

    class_names = load_class_names()
    if not class_names:
        st.stop()

    processed_image = preprocess_image(image, target_size=(224, 224))

    with st.spinner("🔍 Making prediction..."):
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = np.max(predictions[0]) * 100

        if predicted_class.lower() == "good" and confidence > 99:
            st.success(f"✅ Prediction: **{predicted_class}** with confidence of **{confidence:.2f}%**")
        else:
            st.error(f"⚠️ Prediction: **{predicted_class}** with confidence of **{confidence:.2f}%**")
