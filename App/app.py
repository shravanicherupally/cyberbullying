from flask import Flask, render_template, request
import joblib
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load best model and vectorizer
model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# For platform encoding — must match training!
platform_encoder = LabelEncoder()
platform_encoder.fit(['twitter', 'facebook', 'instagram', 'youtube', 'reddit'])

# Text cleaning (must match training!)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        comment = request.form['comment']
        platform = request.form['platform'].lower().strip()

        # Clean comment
        cleaned = clean_text(comment)
        comment_vec = vectorizer.transform([cleaned])

        # Encode platform
        if platform in platform_encoder.classes_:
            platform_encoded = platform_encoder.transform([platform])[0]
        else:
            platform_encoded = 0  # default/fallback if unknown

        # Combine features
        from scipy.sparse import hstack
        X_final = hstack((comment_vec, np.array([[platform_encoded]])))

        # Predict
        pred = model.predict(X_final)[0]
        prediction = 'Bullying' if pred == 1 else 'Non-Bullying'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
