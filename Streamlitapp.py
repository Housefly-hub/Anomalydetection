# import streamlit as st # type: ignore
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # Load model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("keras_model.h5")

# # Load class labels
# def load_class_names(path="classes.txt"):
#     with open(path, "r") as f:
#         return [line.strip() for line in f.readlines()]

# # Preprocess uploaded image
# def preprocess_image(image, target_size):
#     image = image.convert("RGB")
#     image = image.resize(target_size)
#     img_array = tf.keras.utils.img_to_array(image)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#     return img_array

# # Streamlit UI
# st.title("🔧 Screw Anomaly Detector")
# st.write("Upload an image of a screw to detect if it is **Good** or **Anomalous**.")

# uploaded_file = st.file_uploader("📤 Upload a screw image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

#     model = load_model()
#     class_names = load_class_names()
#     processed_image = preprocess_image(image, target_size=(224, 224))  # Adjust to your model

#     predictions = model.predict(processed_image)
#     predicted_index = np.argmax(predictions)
#     predicted_label = class_names[predicted_index]
#     confidence = np.max(predictions)

#     if predicted_label == "Good Screw":
#         st.success("✅ This is a **Good Screw**.")
#     else:
#         st.error("⚠️ This screw appears to have an **Anomaly**.")

#         if st.button("📋 About Anomaly"):
#             st.info(f"🔍 Predicted Anomaly Type: **{predicted_label}**")
#             descriptions = {
#                 "Deformed Tip Screw": "The screw tip is bent, blunt, or broken, making it hard to insert.",
#                 "Scratched Head Screw": "Visible surface damage or scratches on the head, possibly affecting aesthetics or fit.",
#                 "Scratched Neck Screw": "Marks or abrasions around the neck of the screw, often from handling or manufacturing defects.",
#                 "Defected Thread Screw": "Thread pattern is incomplete, chipped, or uneven, which may reduce grip or holding power.",
#             }
#             st.markdown(f"**Description:** {descriptions.get(predicted_label, 'No detailed description available.')}")
#             st.markdown("🔁 You can try uploading another image.")


import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("keras_model.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load class labels
def load_class_names(path="classes.txt"):
    try:
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Class names file 'classes.txt' not found.")
        return []
    except Exception as e:
        st.error(f"Error loading class names: {e}")
        return []

# Preprocess uploaded image
def preprocess_image(image, target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizing the image
    return img_array

# Streamlit UI
st.title("🔧 Screw Anomaly Detector")
st.write("Upload an image of a screw to detect if it is **Good** or **Anomalous**.")

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

    processed_image = preprocess_image(image, target_size=(224, 224))  # Adjust to your model

    with st.spinner("Making predictions..."):
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        predicted_label = class_names[predicted_index]
        confidence = np.max(predictions)

    st.write(f"Prediction: **{predicted_label}** with confidence: {confidence:.2f}")

    if predicted_label == "Good Screw":
        st.success("✅ This is a **Good Screw**.")
    else:
        st.error("⚠️ This screw appears to have an **Anomaly**.")

        if st.button("📋 About Anomaly"):
            st.info(f"🔍 Predicted Anomaly Type: **{predicted_label}**")
            descriptions = {
                "Deformed Tip Screw": "The screw tip is bent, blunt, or broken, making it hard to insert.",
                "Scratched Head Screw": "Visible surface damage or scratches on the head, possibly affecting aesthetics or fit.",
                "Scratched Neck Screw": "Marks or abrasions around the neck of the screw, often from handling or manufacturing defects.",
                "Defected Thread Screw": "Thread pattern is incomplete, chipped, or uneven, which may reduce grip or holding power.",
            }
            st.markdown(f"**Description:** {descriptions.get(predicted_label, 'No detailed description available.')}")

        st.markdown("🔁 You can try uploading another image.")
