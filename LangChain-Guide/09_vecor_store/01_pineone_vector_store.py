from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv


# Load all ENV Variables
load_dotenv()

# ========================== Load Data through PDF ==========================
file_path = "E:/What Are AI Agents.pdf  "
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
            metadata = {"source": "What is Ai Agents.pdf",
                        "page_no": page.metadata['page']+1}

            # Create a Document object with content and metadata
            doc_string = Document(page_content=pg_sub_split, metadata=metadata)

            # Append the Document object to the list
            docs.append(doc_string)
    except Exception as e:
        print(f"Error processing page {
              page.metadata.get('page', 'unknown')}: {e}")

print(len(docs))

# ========================== Create embedding and save to the Vector Store ==========================

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
index_name = "ai-agents"

pinecone_vector_db = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddings
)


# ========================== Retrive data in our Data Base ==========================

# Define a query to search for relevant information
query = "What is AI Agents?"

# Perform similarity search to find the top 5 most relevant results
pinecone_results = pinecone_vector_db.similarity_search(query, k=5)

print(len(pinecone_results))

print(pinecone_results[0])
print(pinecone_results[1])
print(pinecone_results[2])
print(pinecone_results[3])
print(pinecone_results[4])