import joblib

# Load trained model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Simple list of bullying-related words
bully_keywords = [
    "bad", "stupid", "idiot", "fool", "ugly", "hate", "kill", "loser",
    "waste", "shit", "nonsense", "donkey", "dog", "poda", "loosu"
]

# Keyword detection
def is_keyword_bullying(text):
    words = text.lower().split()
    return any(word in words for word in bully_keywords)

# Hybrid prediction function
def predict_text(text):
    # Check keywords first
    if is_keyword_bullying(text):
        return "Cyberbullying Detected (Keyword Match)"

    # ML model prediction
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]

    if prediction == 1:  # 1 = bullying
        return "Cyberbullying Detected (ML Model)"
    else:
        return "No Cyberbullying Detected"


# CLI interaction loop
print("=== Cyberbullying Detection System ===")

while True:
    user_text = input("\nEnter any text:\n")
    
    if user_text.lower() in ["exit", "quit"]:
        print("Exiting system...")
        break

    result = predict_text(user_text)
    print("\n➡️ Result:", result)
