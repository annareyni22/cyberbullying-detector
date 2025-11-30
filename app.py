import streamlit as st
import joblib

# Load model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# List of strong offensive words
offensive_words = [
    "fuck", "f***", "bitch", "asshole", "bastard", "slut", "hoe",
    "moron", "idiot", "kill yourself", "stupid", "retard", "dumb"
]

st.title("🔐 Cyberbullying Detection System")
st.write("Enter any text and the system will classify it.")

user_text = st.text_area("Enter Text")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_lower = user_text.lower()

        # 1️⃣ Rule-based check
        if any(word in text_lower for word in offensive_words):
            st.error("🚨 Cyberbullying Detected! (Rule-based)")
        
        else:
            # 2️⃣ ML prediction
            text_tfidf = vectorizer.transform([user_text])
            prediction = model.predict(text_tfidf)[0]

            if prediction == 1:
                st.error("🚨 Cyberbullying Detected! (ML Model)")
            else:
                st.success("✅ No Cyberbullying Detected.")
