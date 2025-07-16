from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
# import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
with open('email_spam_detection_model', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer_spam', 'rb') as file:
    vectorizer = pickle.load(file)

# @app.route('/')
# def index():
#     return render_template('templates/email_spam.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('message')
    if not data:
        return jsonify({'error': 'No message provided'}), 400

    message_tfidf = vectorizer.transform([data])
    prediction = model.predict(message_tfidf)[0]
    prob_spam = model.predict_proba(message_tfidf)[0][1]
    label = 'Spam' if prediction == 1 else 'Ham'
    
    return jsonify({'label': label, 'probability': round(prob_spam * 100, 2)})

if __name__ == '__main__':
    app.run(debug=True)
