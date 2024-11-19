
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load the USE model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load and preprocess dataset
file_path = 'mentalhealth.csv'  # Make sure this file is uploaded
df = pd.read_csv(file_path)
questions = df['Questions'].values
answers = df['Answers'].values
question_embeddings = use_model(questions).numpy()

def get_embeddings(texts):
    return use_model(texts).numpy()

def find_best_answer(user_question):
    user_embedding = get_embeddings([user_question])
    similarities = cosine_similarity(user_embedding, question_embeddings).flatten()
    best_match_index = np.argmax(similarities)
    return answers[best_match_index], similarities[best_match_index]

@app.route('/get_answer', methods=['POST'])
def get_answer():
    user_question = request.json.get('question')
    best_answer, similarity_score = find_best_answer(user_question)
    return jsonify({'answer': best_answer, 'similarity': float(similarity_score)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
