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
file_path = "E:/AI Policy.pdf"
data_loaders = PyPDFLoader(file_path)

try:
    # Extract pages from the loaded PDF
    pages = data_loaders.load()
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
            metadata = {"source": "National AI Policy",
                        "page_no": page.metadata['page']+1}

            # Create a Document object with content and metadata
            doc_string = Document(page_content=pg_sub_split, metadata=metadata)

            # Append the Document object to the list
            docs.append(doc_string)
    except Exception as e:
        print(f"Error processing page {
              page.metadata.get('page', 'unknown')}: {e}")

print(len(docs))

# ========================== Initialize Qdrant and Create Collection ==========================

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

qdrant_url = "https://d6717ddf-9785-4505-8417-d5a38b843fa8.us-east4-0.gcp.cloud.qdrant.io:6333/"
qdrant_key = os.getenv("QDRANT_API_KEY")
collection_name = "AI-Policy"

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
    vectorstore_from_docs = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_key,
        collection_name=collection_name  # Explicitly specify the collection name
    )
    print("Data successfully stored in Qdrant.")
except Exception as e:
    print("Error storing data in Qdrant:", e)

question = "What is AI policy for students?"

# ========================== Drawbacks of Similarity Search returns the same chunks ==========================

print("========================== Similarity Search ================================")
docs_ss = vectorstore_from_docs.similarity_search(question, k=5)

for response_ss in docs_ss:
    print("==========================================================")
    print(response_ss)

# ========================== Solves overlap problem and to enforce diversity in search results ==========================

print("=========================== Maximum marginal relevance ===============================")
docs_mmr = vectorstore_from_docs.max_marginal_relevance_search(question, k=5)

for response_mmr in docs_mmr:
    print("==========================================================")
    print(response_mmr)

# ========================== Solves overlap problem and to enforce diversity in search results ==========================

# Change the filter key to match the metadata 'page_no' in the documents
docs_md = vectorstore_from_docs.similarity_search(
    question,
    k=3,
    filter={
        "must": [
            {
                "key": "page_no",  # Match the 'page_no' metadata key
                "match": {
                    "value": 7  # Replace with a page number that exists in your data
                }
            }
        ]
    }
)

# Print the retrieved documents
if docs_md:
    for doc in docs_md:
        print("Page No:", doc.metadata['page_no'])
        print("Content:", doc.page_content)
        print("\n---\n")
else:
    print("No documents found for the specified filter.")

# https://qdrant.tech/documentation/concepts/filtering/