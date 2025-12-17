from typing import List
from llama_index.core import VectorStoreIndex
from app.core.milvus_client import get_vector_store, get_embedding_model
from app.models.schemas import SearchResponse

def search_faq(query: str, top_k: int = 5) -> List[SearchResponse]:
    vector_store = get_vector_store()
    
    # Load index from Milvus
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=get_embedding_model()
    )
    
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    
    results = []
    for node in nodes:
        # Extract metadata
        question = node.metadata.get("question", "N/A")
        answer = node.metadata.get("answer", "N/A")
        score = node.score if node.score is not None else 0.0
        
        results.append(SearchResponse(
            question=question,
            answer=answer,
            score=score
        ))
        
    return results
