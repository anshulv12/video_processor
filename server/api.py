from flask import Flask, request, jsonify
from embeddings import Embedder
from downloader import Downloader

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
        embedder = Embedder()
    results = embedder.find_relevant_urls(query)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)