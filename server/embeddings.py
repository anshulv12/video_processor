import os
from pinecone import Pinecone
from uuid import uuid4

class Embedder:
    def __init__(self, index_name: str = "video-test-anshul"):
        pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
        if not pc.has_index(index_name):
            pc.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model": "llama-text-embed-v2",
                }
            )
        self.index = pc.index(index_name)
    
    def embed_text_url(self, metadata, url, namespace="main"):
        vector = [
            {
                "_id": f'rec_{uuid4()}',
                "text": metadata,
                "url": url
            }
        ]
        self.index.upsert(vectors=vector, namespace=namespace)

    def find_relevant_urls(self, query, namespace="main", similarity_threshold=0.7):
        response = self.index.search(
            namespace=namespace, 
            query={
                "inputs": {"text": query}, 
                "top_k": 100
            },
            fields=['text', 'url']
        )
        filtered_results = [result['fields'] for result in response['result']['hits'] if result['score'] >= similarity_threshold]
        return filtered_results