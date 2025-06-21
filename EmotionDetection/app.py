import streamlit as st
import joblib
import os
import os
import streamlit as st

st.write("📂 Current directory:", os.getcwd())
st.write("📄 Files:", os.listdir())

# Load trained model and vectorizer
model = joblib.load('emotion_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Emoji → emotion (for emoji-only input)
emoji_to_emotion = {
    "😂": "joy",
    "😊": "happy",
    "😍": "joy",
    "😢": "sadness",
    "😭": "sadness",
    "😱": "surprise",
    "😨": "fear",
    "😐": "neutral",
    "🤢": "disgust",
    "🤮": "disgust",
    "😡": "anger",
    "😠": "anger",
    "❤️": "joy"
}

# Emotion → emoji (for output)
emotion_to_emoji = {
    "joy": "😂",
    "happy": "😊",
    "sadness": "😢",
    "surprise": "😱",
    "fear": "😨",
    "neutral": "😐",
    "disgust": "🤢",
    "anger": "😡"
}

# Function to detect emoji-based emotion if present
def detect_from_emoji(text):
    for emoji, emotion in emoji_to_emotion.items():
        if emoji in text:
            return emotion
    return None

# Streamlit UI setup
st.set_page_config(page_title="Emoji-Based Emotion Detection", layout="centered")
st.title("🧠 Emotion Detection with Emoji 🧡")
st.markdown("Type a sentence or just use emojis to get the emotion with an emoji response!")

# User input
user_input = st.text_input("✍️ Enter sentence or emoji:")

# Predict button
if st.button("🔍 Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence or emoji.")
    else:
        # First check for emojis
        emoji_based_emotion = detect_from_emoji(user_input)

        if emoji_based_emotion:
            emoji = emotion_to_emoji.get(emoji_based_emotion, "❓")
            st.success(f"🎯 Detected from Emoji: **{emoji_based_emotion.capitalize()}** {emoji}")
        else:
            try:
                # Predict from ML model
                transformed = vectorizer.transform([user_input])
                model_prediction = model.predict(transformed)[0]
                emoji = emotion_to_emoji.get(model_prediction.lower(), "❓")
                st.success(f"🔍 Predicted by Model: **{model_prediction.capitalize()}** {emoji}")
            except Exception as e:
                st.error(f"❌ Error in prediction: {e}")

# Footer
st.markdown("---")
st.caption("🚀 Made with 💖 for Microsoft-AICTE Internship 2025")
