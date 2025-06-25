from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

# Load all ENV Variables
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# ========================== Load Data through PDF ==========================
file_path = "E:/What Are AI Agents.pdf"
data_loader = PyPDFLoader(file_path)

try:
    # Extract pages from the loaded PDF
    pages = data_loader.load()
    print("Number of pages loaded:", len(pages))
except Exception as e:
    print("Error loading PDF:", e)

# ========================== Create Data Chunks ==========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

# Create an empty list to store processed document chunks
docs = []

# Iterate over each page in the extracted pages
for page in pages:
    try:
        # Split the page content into smaller chunks
        sp_page = text_splitter.split_text(page.page_content)

        # Iterate over each chunk and create Document objects
        for pg_sub_split in sp_page:
            # Metadata for each chunk, including source and page number
            metadata = {
                "source": "What Are AI Agents.pdf",
                "page_no": page.metadata.get('page', 0) + 1
            }

            # Create a Document object with content and metadata
            doc_string = Document(page_content=pg_sub_split, metadata=metadata)

            # Append the Document object to the list
            docs.append(doc_string)
    except Exception as e:
        print(f"Error processing page {
              page.metadata.get('page', 'unknown')}: {e}")

print("Total chunks created:", len(docs))

# ========================== Initialize Qdrant and Create Collection ==========================

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

qdrant_url = "https://d6717ddf-9785-4505-8417-d5a38b843fa8.us-east4-0.gcp.cloud.qdrant.io:6333/"
# Ensure your Qdrant API key is set in environment variables
qdrant_key = os.getenv("QDRANT_API_KEY")    
collection_name = "What_Are_AI_Agents"

try:
    # Initialize Qdrant client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    # Create the collection if it doesn't already exist
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Collection '{collection_name}' created successfully.")

except Exception as e:
    print("Error creating Qdrant collection:", e)

# ========================== Store Data in Qdrant ==========================

try:
    # Initialize Qdrant Vector Store
    qdrant = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_key,
        collection_name=collection_name  # Explicitly specify the collection name
    )
    print("Data successfully stored in Qdrant.")
except Exception as e:
    print("Error storing data in Qdrant:", e)

# Define a query to search for relevant information
query = "What is AI Agents?"

# ========================== Retrive data in our Data Base ==========================

# Perform similarity search to find the top 5 most relevant results
qdrant_results = qdrant.similarity_search(query, k=5)

print(len(qdrant_results))

print(qdrant_results[0])
print(qdrant_results[1])
print(qdrant_results[2])
print(qdrant_results[3])
print(qdrant_results[4])