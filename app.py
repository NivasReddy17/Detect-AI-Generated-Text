from flask import Flask, request, jsonify, render_template
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# custom tokenizer function
def custom_tokenizer(text):
    return re.findall(r'\b\w+\b', text)

# Loading ensemble model and vectorizer
try:
    with open('ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

def preprocess_text(text):
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['text']
        print(f"Received text: {data}")  # Debugging
        normalized_text = preprocess_text(data)
        print(f"Normalized text: {normalized_text}")  # Debugging
        vectorized_text = vectorizer.transform([normalized_text])
        proba = model.predict_proba(vectorized_text)[0][1]
        print(f"Predicted probability: {proba}")  # Debugging
        return jsonify({'probability': proba})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
