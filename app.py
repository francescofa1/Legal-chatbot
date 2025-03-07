from flask import Flask, render_template, request, jsonify
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from utils import load_openai_api_key
import os

app = Flask(__name__)

api_key = load_openai_api_key()

def setup_index():
    docs = SimpleDirectoryReader(input_dir="documents").load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index

index = setup_index()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message", "")
    response = index.query(user_input).response
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
