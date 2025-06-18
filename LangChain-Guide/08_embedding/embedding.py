# Import necessary libraries
from dotenv import load_dotenv                                       # For loading environment variables from a .env file
from langchain_google_genai import GoogleGenerativeAIEmbeddings      # For using Google Generative AI Embeddings
from langchain_huggingface import HuggingFaceEmbeddings              # For using HuggingFace Embeddings

# Load environment variables from a .env file (used for API keys or other configuration)
load_dotenv()

# ================================================ Hugging Face Embedding model ================================================

# Initialize the Hugging Face embedding model with a specific pre-trained model
huggingface_embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5")  # Using BAAI's 'bge-small-en-v1.5' model

# Define the input text to generate embeddings
text = "What is AI & tell me the benefits of AI"

# Generate the embeddings for the input text using the Hugging Face model
text_embedding = huggingface_embedding_model.embed_query(text)

# Print the length of the generated embedding and the embedding vector itself
print("==================== Hugging Face Embedding Model ==========================")
print(len(text_embedding))  # Outputs the length of the embedding vector
print(text_embedding)  # Outputs the actual embedding vector

# ================================================ Google Embedding model ================================================

# Initialize the Google Generative AI embedding model with a specific model version
google_embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004")  # Using Google's 'text-embedding-004' model

# Define the input text to generate embeddings
text = "What is AI & tell me the benefits of AI"

# Generate the embeddings for the input text using the Google model
embed = google_embedding_model.embed_query(text)

# Print the length of the generated embedding and the embedding vector itself
print("==================== Google Embedding Model ==========================")
print(len(embed))  # Outputs the length of the embedding vector
print(embed)  # Outputs the actual embedding vector