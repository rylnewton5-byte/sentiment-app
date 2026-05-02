import streamlit as st
from transformers import pipeline

# Load Hugging Face sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

classifier = load_model()

# UI
st.title("🤖 AI Sentiment Analysis (Hugging Face)")

text = st.text_area("Enter your text:")

if st.button("Analyze"):
    if text.strip() != "":
        result = classifier(text)[0]

        label = result['label']
        score = result['score']

        if label == "POSITIVE":
            st.success(f"😊 Positive (Confidence: {score:.2f})")
        else:
            st.error(f"😡 Negative (Confidence: {score:.2f})")
    else:
        st.warning("Please enter some text.")