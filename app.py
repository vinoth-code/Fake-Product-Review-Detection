import streamlit as st
import pickle
import re
import os
import gdown

# URLs
model_id = "1DsuW9X1V8_rz2YnJwtfa6TA4pcRgWvMs"
vectorizer_id = "1PwLwgrs16y6rSZ7J0rrWlbpXWm5e5WCM"

# Output paths
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

# Download model if not already downloaded
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={model_id}", model_path, quiet=False)

if not os.path.exists(vectorizer_path):
    gdown.download(f"https://drive.google.com/uc?id={vectorizer_id}", vectorizer_path, quiet=False)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Clean text (must match training)
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text))
    return text.lower()

# App layout
st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("🕵️ Fake Product Review Detector")
st.markdown("### 📝 Enter your review below to detect if it's **Fake** or **Genuine**")

# Input
user_input = st.text_area("Enter a product review:")

# Button
if st.button("🔍 Detect"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = round(max(probability) * 100, 2)

        if prediction == 1:
            st.success("✅ This review is **Genuine**.")
        else:
            st.error("❌ This review is **Fake**.")

        st.info(f"Model confidence: **{confidence}%**")
