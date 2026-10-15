"""
Brain Tumor MRI Classifier - Streamlit app
Run:  streamlit run app.py
Needs: ./models/best_model.h5 and ./models/class_names.json (created by the notebook).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_DIR = "models"
LOW_CONFIDENCE = 0.75

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str):
    # Normalisation layers are part of the saved model, so raw 0-255 pixels can be passed directly.
    return keras.models.load_model(path, compile=False)


@st.cache_data
def load_class_names():
    with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
        return json.load(f)


def predict(model, image: Image.Image) -> np.ndarray:
    h, w = model.input_shape[1:3]
    img = image.convert("RGB").resize((w, h))
    batch = np.expand_dims(np.asarray(img, dtype="float32"), 0)
    return model.predict(batch, verbose=0)[0]


# ----------------------------------------------------------------------------- UI
st.title("🧠 Brain Tumor MRI Classifier")
st.caption("Upload a brain MRI image and the model will predict the tumor type with a confidence score.")

model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.h5")))
if not model_files or not os.path.exists(os.path.join(MODEL_DIR, "class_names.json")):
    st.error("No trained model found. Run the notebook first so that the `models/` folder "
             "(with `best_model.h5` and `class_names.json`) is created next to this file.")
    st.stop()

names = [os.path.basename(p) for p in model_files]
default = names.index("best_model.h5") if "best_model.h5" in names else 0
with st.sidebar:
    st.header("Settings")
    chosen = st.selectbox("Model", names, index=default,
                          help="best_model.h5 is the model selected in the notebook. Others are for comparison.")
    st.markdown("---")
    st.markdown("**How to use**\n1. Upload an MRI image (JPG/PNG)\n2. Read the prediction and confidence\n3. Check the probability of every class")

model = load_model(os.path.join(MODEL_DIR, chosen))
class_names = load_class_names()

uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is None:
    st.info("👆 Upload an MRI image to get a prediction.")
else:
    try:
        image = Image.open(uploaded)
    except Exception:
        st.error("This file could not be read as an image. Please upload a valid JPG/PNG.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded MRI", use_container_width=True)

    with st.spinner("Analysing..."):
        probs = predict(model, image)

    top = int(probs.argmax())
    conf = float(probs[top])
    with col2:
        st.subheader("Prediction")
        st.metric("Tumor type", class_names[top].replace("_", " ").title())
        st.metric("Confidence", f"{conf:.1%}")
        if conf < LOW_CONFIDENCE:
            st.warning("Low confidence - this scan should be reviewed carefully by a specialist.")

    st.subheader("Probability of each class")
    prob_df = pd.DataFrame({"Class": [c.replace("_", " ").title() for c in class_names],
                            "Probability": probs}).set_index("Class")
    st.bar_chart(prob_df)
    st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)

st.markdown("---")
st.caption("⚠️ Educational project. This tool is an AI aid and NOT a medical diagnosis. "
           "Always consult a qualified radiologist or doctor.")
