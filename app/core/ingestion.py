from typing import List
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from app.models.schemas import FAQItem
from app.core.milvus_client import get_vector_store, get_embedding_model

def ingest_faqs(faqs: List[FAQItem]):
    """
    Ingest FAQs into the vector store.
    
    Each FAQ is converted into a Document with combined question and answer text.
    The documents are then processed with a SentenceSplitter (semantic splitting + overlap)
    and indexed into Milvus.
    
    Args:
        faqs: List of FAQItem objects to be ingested.
        
    Returns:
        int: The number of documents processed.
    """
    documents = []
    for faq in faqs:
        # We combine question and answer for context, but metadata stores them separately
        text = f"Question: {faq.question}\nAnswer: {faq.answer}"
        doc = Document(
            text=text,
            metadata={
                "question": faq.question,
                "answer": faq.answer
            }
        )
        documents.append(doc)

    # Semantic splitting + overlap
    # We use SentenceSplitter to ensure semantic integrity of chunks.
    # chunk_size=512 ensures reasonable context length.
    # chunk_overlap=50 ensures continuity between chunks.
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # This will automatically split documents using the parser if they are too long
    # and index them into Milvus using the configured embedding model.
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[parser],
        embed_model=get_embedding_model()
    )
    
    return len(documents)
