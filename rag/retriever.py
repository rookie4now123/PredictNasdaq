from llama_index.core import Document, VectorStoreIndex

def build_index():
    docs = [
        Document(text="Tech stocks rally due to strong earnings."),
        Document(text="Market uncertainty due to geopolitical tensions."),
        Document(text="Federal Reserve signals interest rate cuts.")
    ]
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine()