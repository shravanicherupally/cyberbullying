import joblib
import re
import numpy as np
from scipy.sparse import hstack

from sklearn.preprocessing import LabelEncoder

# ✅ FIXED PATH!
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Same platform encoder as training
platform_encoder = LabelEncoder()
platform_encoder.fit(['twitter', 'facebook', 'instagram', 'youtube', 'reddit'])

# Same cleaning as preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# 🟢 Test 1: Non-Bullying
test_comment_1 = "I hope you have a wonderful day!"
platform_1 = "twitter"

# 🟢 Test 2: Bullying
test_comment_2 = "You are so dumb, everyone hates you!"
platform_2 = "twitter"

for comment, platform in [(test_comment_1, platform_1), (test_comment_2, platform_2)]:
    cleaned = clean_text(comment)
    vec = vectorizer.transform([cleaned])
    platform_encoded = platform_encoder.transform([platform.lower()])[0]
    X_final = hstack((vec, np.array([[platform_encoded]])))

    prediction = model.predict(X_final)[0]
    label = "Bullying" if prediction == 1 else "Non-Bullying"
    print(f"Comment: {comment}")
    print(f"Prediction: {prediction} ({label})")
    print("="*50)
