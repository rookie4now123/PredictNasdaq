from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
def build_index():
    # ✅ Force local embedding model (NO OpenAI)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash-lite", temperature=0)
    docs = [
        Document(text="Tech stocks rally due to strong earnings."),
        Document(text="Market uncertainty due to geopolitical tensions."),
        Document(text="Federal Reserve signals interest rate cuts.")
    ]

    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine()