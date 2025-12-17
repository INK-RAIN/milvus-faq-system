from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings as LlamaSettings
from app.core.config import settings

def get_embedding_model():
    return HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)

def get_vector_store():
    return MilvusVectorStore(
        uri=settings.MILVUS_URI,
        collection_name=settings.COLLECTION_NAME,
        dim=settings.DIMENSION,
        overwrite=False
    )

def init_settings():
    LlamaSettings.embed_model = get_embedding_model()
