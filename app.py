from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/', methods=['GET'])
def home():
    return '''
        <h2>✅ Semantic Similarity API is Running!</h2>
        <p>Use <code>POST /predict</code> with JSON body:<br>
        {"text1": "your first sentence", "text2": "your second sentence"}</p>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text1 = data.get("text1", "")
    text2 = data.get("text2", "")
    if not text1 or not text2:
        return jsonify({"error": "Both text1 and text2 are required"}), 400

    embeddings = model.encode([text1, text2])
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return jsonify({"similarity score": round(score, 3)})
