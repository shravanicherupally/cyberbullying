from flask import Flask, render_template, request
import joblib
import re
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

platform_encoder = LabelEncoder()
platform_encoder.fit(['twitter', 'facebook', 'instagram', 'youtube', 'reddit'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    comment = ''
    platform = ''
    if request.method == 'POST':
        comment = request.form['comment']
        platform = request.form['platform']

        cleaned = clean_text(comment)
        vec = vectorizer.transform([cleaned])
        platform_encoded = platform_encoder.transform([platform.lower()])[0]
        X_final = hstack((vec, np.array([[platform_encoded]])))
        pred = model.predict(X_final)[0]
        prediction = "Bullying" if pred == 1 else "Non-Bullying"

    return render_template(
        'index.html',
        prediction=prediction,
        comment=comment,
        platform=platform
    )

if __name__ == "__main__":
    app.run(debug=True)
