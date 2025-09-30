import re
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------ SETTINGS ------------------ #
MODEL_PATH = "sentiment_LSTM_model.keras"   # change to .h5 if needed
TOKENIZER_PATH = "tokenizer.pickle"
MAX_SEQUENCE_LENGTH = 200

# ------------------ MODEL LOADER ------------------ #
def load_sentiment_model(path):
    """Load a trained sentiment model safely with fallback."""
    model = None
    try:
        # Try direct load (works if Keras/TensorFlow versions match)
        model = load_model(path, compile=False)
        st.success(f"✅ Model loaded directly: {path}")
    except Exception as e:
        st.warning(f"⚠️ Direct load failed: {e}")

        # Fallback: rebuild architecture and load weights
        if path.endswith(".h5") or path.endswith(".keras"):
            st.info("🔄 Rebuilding architecture and loading weights...")
            # ⚠️ Must match Colab architecture exactly!
            model = Sequential([
                Embedding(input_dim=10000, output_dim=128, input_length=MAX_SEQUENCE_LENGTH),
                LSTM(128),
                Dropout(0.5),
                Dense(1, activation="sigmoid")
            ])
            try:
                model.load_weights(path)
                st.success("✅ Weights loaded into rebuilt model.")
            except Exception as inner_e:
                st.error(f"❌ Could not load weights: {inner_e}")
                model = None
    return model

# ------------------ TOKENIZER LOADER ------------------ #
@st.cache_resource
def load_tokenizer(path):
    with open(path, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# ------------------ TEXT PREPROCESSOR ------------------ #
def preprocess_text(text, tokenizer, max_len=MAX_SEQUENCE_LENGTH):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove non-alphanumeric
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded

# ------------------ PREDICT FUNCTION ------------------ #
def predict_sentiment(model, tokenizer, review):
    processed = preprocess_text(review, tokenizer)
    prediction = model.predict(processed)
    score = float(prediction[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, score

# ------------------ STREAMLIT APP ------------------ #
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review and predict whether it is **Positive** or **Negative**.")

# Load model and tokenizer
model = load_sentiment_model(MODEL_PATH)
tokenizer = load_tokenizer(TOKENIZER_PATH)

# User input
user_input = st.text_area("✍️ Movie Review")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a review first.")
    elif model is None:
        st.error("❌ Model not available. Please check loading.")
    else:
        sentiment, confidence = predict_sentiment(model, tokenizer, user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.2f}")

# if sentiment == "Positive":
#     st.balloons()   # celebration for positive sentiment
# else:
#     st.snow()       # snow effect for negative sentiment
    
