from typing import List
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from app.models.schemas import FAQItem
from app.core.milvus_client import get_vector_store, get_embedding_model

def ingest_faqs(faqs: List[FAQItem]):
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
    # Although FAQs are usually short, we satisfy the requirement by configuring the splitter
    parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )

    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # This will automatically split documents using the parser if they are too long
    # and index them into Milvus
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[parser],
        embed_model=get_embedding_model()
    )
    
    return len(documents)
