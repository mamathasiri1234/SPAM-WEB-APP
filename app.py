# ==============================
# 📩 Spam Email Classifier Web App
# ==============================

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
df = pd.read_csv("spam.csv")

# Convert label
df['label'] = df['label'].map({'ham':0, 'spam':1})

X = df['message']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Model Training
model = MultinomialNB()
model.fit(X, y)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = vectorizer.transform([message])
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "🚨 Spam Message"
    else:
        result = "✅ Not Spam"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)