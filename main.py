from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Loading model from local folder
model = SentenceTransformer("./all-minilm-finetuned/checkpoint-471")

# Some sample documents (knowledge base)
documents = [
    "The sun is the closest star to Earth and provides the energy that sustains life.",
    "The moon orbits the Earth approximately every 27 days.",
    "Python is a high-level programming language popular in AI, data science, and web development.",
    "JavaScript is mainly used for web development to make websites interactive.",
    "The 8086 microprocessor was introduced by Intel in 1978.",
    "The World Wide Web was invented by Tim Berners-Lee in 1989.",
    "Mahatma Gandhi led India’s independence movement through non-violent resistance.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Mount Everest, located in the Himalayas, is the tallest mountain in the world.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The human brain contains approximately 86 billion neurons.",
    "The Great Wall of China is one of the Seven Wonders of the World.",
    "Albert Einstein proposed the theory of relativity in the early 20th century.",
    "The Earth takes 365.25 days to complete one orbit around the Sun.",
    "DNA is the molecule that carries genetic information in living organisms.",
    "The Amazon Rainforest is often referred to as the lungs of the Earth.",
    "The Internet is a global network connecting millions of computers worldwide.",
    "Bitcoin is the first decentralized cryptocurrency, introduced in 2009.",
    "Soccer (football) is the most widely played and watched sport in the world."
]


# Encode documents
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# Creating FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Function to retrieve best match
def rag_query(query, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=top_k)
    results = [documents[i] for i in indices[0]]
    return results


print(" Fine-tuned RAG Bot is ready! Type 'exit' to quit.\n")
while True:
    user_query = input("You: ")
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Bot: Goodbye! ")
        break
    
    results = rag_query(user_query, top_k=1)  
    print("Bot:", results[0])
    print()
