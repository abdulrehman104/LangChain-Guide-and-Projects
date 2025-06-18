from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def load_pdf(file_path):
    """Loads a PDF file and returns the extracted pages."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"Total pages loaded: {len(pages)}")
    return pages


def split_text(pages, chunk_size=1000, chunk_overlap=200):
    """Splits text into chunks for better embedding processing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.split_documents(pages)
    print(f"Total document chunks created: {len(docs)}")
    return docs


def create_vector_store(docs):
    """Creates a Chroma vector store and adds embeddings."""

    index_name = "chat-with-data"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = PineconeVectorStore.from_documents(
        documents=docs,
        index_name=index_name,
        embedding=embeddings,
    )

    print(f"Vector store created with {len(docs)} embeddings.")
    return vector_store


if __name__ == "__main__":
    file_path = "https://drive.google.com/uc?export=download&id=1H3sqzQtcTvIQqoRjyfG9kfLmGuMBLwDW"
    pages = load_pdf(file_path)
    docs = split_text(pages)
    vector_store = create_vector_store(docs)


# def similarity_search(vector_store, query):
#     """Performs similarity search on the vector database."""
#     docs = vector_store.similarity_search_with_score(query, k=1)

#     if docs:
#         print("Top search result:", docs[0].page_content)
#         return docs[0].page_content
#     return "No relevant results found."


# if __name__ == "__main__":
#     file_path = "https://drive.google.com/uc?export=download&id=1H3sqzQtcTvIQqoRjyfG9kfLmGuMBLwDW"
#     pages = load_pdf(file_path)
#     docs = split_text(pages)
#     vector_store = create_vector_store(docs)
#     query = "What is Generative AI?"
#     result = similarity_search(vector_store, query)
#     print("Search result:", result)
