from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import chromadb
import spacy
from chromadb.utils import embedding_functions
import pandas as pd
import json

app = Flask(__name__)

# Initialize ChromaDB client and collections
EMBED_MODEL = "all-MiniLM-L6-v2"
client = chromadb.PersistentClient(path="vectordata")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
collection = client.get_or_create_collection(name='combined_data', embedding_function=embedding_func, metadata={"hnsw:space": "cosine"})
sym_collection = client.get_or_create_collection(name='symptoms_data', embedding_function=embedding_func, metadata={"hnsw:space": "cosine"})


nlp = spacy.load('en_core_web_sm')

def preprocess_symptom(symptom_text):
    sym = nlp(symptom_text)
    cleaned_sym = [words.lemma_.lower() for words in sym if words.text.lower() not in spacy.lang.en.stop_words.STOP_WORDS and words.text.isalpha()]
    return ' '.join(cleaned_sym)

def chroma_query(query):
    query = query.lower()
    query = preprocess_symptom(query)
    result = collection.query(query_texts=query, n_results=3, include=["documents", "distances"])
    result1 = sym_collection.query(query_texts=query, n_results=3, include=["documents","distances"])
    return result, result1

def process_chroma_query_result(result):
    output = []
    identified_symptoms = result[1]['documents'][0]
    output.append({'identified symptoms': identified_symptoms})
    disease_info = result[0]['documents'][0]
    distances = result[0]['distances'][0]
    for i, info in enumerate(disease_info):
        disease_data = {}
        info_parts = info.split('|')
        for part in info_parts:
            key, value = part.split(':', 1)
            if key == 'disease':
                disease_data[key] = value
            elif key == 'symptoms':
                disease_data['symptom'] = value
            elif key == 'etiology':
                disease_data[key] = value
        disease_data['confidence'] = 1 - distances[i]
        output.append(disease_data)
    return output

@app.route('/nlp_predict', methods=['POST'])
def predict():
    data = request.json
    query = data.get('query', '')
    result = chroma_query(query)
    output = process_chroma_query_result(result)
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
