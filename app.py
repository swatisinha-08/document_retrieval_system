from flask import Flask, request, jsonify
import redis
import pymongo
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
import threading
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='api.log', level=logging.INFO)

# MongoDB setup
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
db = mongo_client["document_db"]
document_collection = db["documents"]

# Redis setup
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index setup
index = faiss.IndexFlatL2(384)  # Dimensions should match the embedding size

# Function to calculate embedding
def get_embedding(text):
    return model.encode([text])[0]

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running"})

@app.route("/search", methods=["POST"])
def search():
    user_id = request.json.get("user_id")
    text = request.json.get("text")
    top_k = int(request.json.get("top_k", 5))
    threshold = float(request.json.get("threshold", 0.7))

    # Log the user's request time
    logging.info(f"User {user_id} requested search at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Track user requests using Redis
    if redis_client.exists(user_id):
        request_count = int(redis_client.get(user_id))
        if request_count > 5:
            logging.info(f"User {user_id} exceeded the request limit.")
            return jsonify({"error": "Too many requests"}), 429
        redis_client.incr(user_id)
    else:
        redis_client.set(user_id, 1)

    start_time = time.time()

    # Convert query text to embedding
    query_embedding = get_embedding(text)

    # Perform FAISS search
    D, I = index.search(np.array([query_embedding]), top_k)
    results = []
    for i, dist in zip(I[0], D[0]):
        if dist < threshold:
            doc = document_collection.find_one({"_id": i})
            results.append({"document": doc['text'], "similarity_score": dist})

    inference_time = time.time() - start_time

    # Log inference time
    logging.info(f"Inference time for user {user_id}: {inference_time:.4f} seconds")

    return jsonify(results)

# Function to scrape news articles
def scrape_news_articles():
    while True:
        articles = requests.get("https://news.ycombinator.com/").text
        soup = BeautifulSoup(articles, 'html.parser')

        for item in soup.find_all('a', class_='storylink'):
            title = item.get_text()
            embedding = get_embedding(title)

            # Insert article into MongoDB
            document_collection.insert_one({"text": title, "embedding": embedding.tolist()})

            # Add embedding to FAISS index
            index.add(np.array([embedding]))

        time.sleep(3600)  # Scrape every hour

# Start scraping in a background thread
thread = threading.Thread(target=scrape_news_articles)
thread.start()

if __name__ == "__main__":
    app.run(debug=True)
