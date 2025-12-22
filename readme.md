# Fine-Tuned RAG Bot (Sentence Transformers + FAISS)

This project implements a simple **Retrieval-Augmented Generation (RAG)** style chatbot using a **fine-tuned SentenceTransformer model** and **FAISS** for semantic search.

The system retrieves the most relevant document from a knowledge base based on user queries.

---

## Files

- **`Fine_tuing_embeddings_model.ipynb`**  
  Used to fine-tune a sentence embedding model and save it locally.

- **`main.py`**  
  Loads the fine-tuned model, creates document embeddings, builds a FAISS index, and runs an interactive query-based retrieval chatbot.

---

## Requirements

```bash
pip install sentence-transformers faiss-cpu numpy tf-keras

