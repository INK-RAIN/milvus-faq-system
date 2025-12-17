import os

class Settings:
    MILVUS_URI = "http://localhost:19530"
    COLLECTION_NAME = "faq_collection"
    DIMENSION = 384  # Dimension for huggingface/all-MiniLM-L6-v2
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"

settings = Settings()
