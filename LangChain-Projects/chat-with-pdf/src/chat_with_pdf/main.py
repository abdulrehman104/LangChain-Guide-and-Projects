import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ========================== Load environment variables from .env file ==========================
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
qdrant_api_key = os.environ.get("QDRANT_API_KEY")
qdrant_url = os.environ.get("QDRANT_URL")

if not gemini_api_key or not qdrant_api_key or not qdrant_url:
    raise ValueError(
        "Please set the GEMINI_API_KEY, QDRANT_API_KEY, and QDRANT_URL environment variables.")


# ========================== Create Funcion to Load PDF ==========================
def load_pdf(file_path: str):
    """Load a PDF file and return its content as a list of documents."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# ========================== Create Funcion that Splits text into chunks ==========================
def split_text_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
    """Split the loaded documents into smaller chunks."""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# ========================== initialize Qdrant Vector Store  ==========================
def initialize_qdrant_collection(collection_name, vector_size):
    """Initializes or recreates a Qdrant collection."""

    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Check if the collection exists
    if qdrant_client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting it...")
        qdrant_client.delete_collection(collection_name)

    # Create a new collection with the correct dimensions
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )


# ========================== Create Embedding and Add this embedding to Vector Store ==========================
def create_vector_store(docs, collection_name):
    """Create a vector store from the documents using Google Generative AI embeddings."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=gemini_api_key)

    # Initialize the Qdrant collection with the correct vector size
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        force_recreate=True  # Ensure the collection is recreated if it exists
    )


# ========================== Create main function and rucall all the pipline function ==========================
def main():
    """Run the complete PDF processing pipeline."""

    print("Starting the PDF processing pipeline...")
    file_path = "https://drive.google.com/uc?export=download&id=1H3sqzQtcTvIQqoRjyfG9kfLmGuMBLwDW"

    # Step 1: Load documents
    print("Loading documents...")
    pages = load_pdf(file_path)
    print(f"Loaded {len(pages)} documents from the URLs.")

    # Step 2: Split documents into chunks
    print("Splitting documents into chunks...")
    docs = split_text_into_chunks(pages)
    print(f"Split the documents into {len(docs)} chunks.")

    # Step 3: Initialize Qdrant collection
    collection_name = "chat_with_pdf"
    vector_size = 768  # GoogleGenerativeAIEmbeddings has 768 dimensions
    print(f"Initializing Qdrant collection '{collection_name}'...")
    initialize_qdrant_collection(collection_name, vector_size)
    print(f"Collection '{collection_name}' initialized successfully.")

    # Step 4: Embed and store documents in Qdrant
    print("Embedding and storing documents in Qdrant...")
    create_vector_store(docs=docs, collection_name="chat_with_pdf")
    print("Documents successfully stored in Qdrant.")
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
