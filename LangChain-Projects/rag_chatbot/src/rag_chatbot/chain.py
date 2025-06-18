import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit app title
st.title("🤖 RAG Chatbot")
st.write("Welcome to the RAG Chatbot! Ask me anything about Generative AI.")

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize Pinecone vector store
index_name = "chat-with-data"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


def create_retriever():
    """Creates a retriever from a vector store."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def create_prompt_template():
    """Creates a well-structured prompt template for better responses."""
    template = """
    You are an AI assistant tasked with answering user questions based on the provided context.
    Read the context carefully and give a well-explained, structured answer in a friendly and professional tone.
    
    If the context does not provide sufficient information, simply say: "I don't have enough information to answer this question."
    
    Try to provide a detailed answer while keeping it easy to understand. Structure your answer in paragraphs and use bullet points if needed.

    Always conclude your response with: "Thanks for asking!"

    Context:
    {context}

    Question: {question}
    
    Answer:
    """
    return ChatPromptTemplate.from_template(template)


def create_rag_chain():
    """Creates a RAG (Retrieval-Augmented Generation) chain with improved processing."""
    retriever = create_retriever()
    prompt = create_prompt_template()

    # Use a better model for detailed responses
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    def retrieve_and_format(query):
        """Retrieves context and formats it properly before passing it to the LLM."""
        retrieved_docs = retriever.invoke(query)

        # Extracting text from retrieved documents
        retrieved_texts = "\n\n".join(
            [doc.page_content for doc in retrieved_docs])

        # Debugging step: print retrieved texts
        print("\n--- Retrieved Context ---\n",
              retrieved_texts, "\n--- End Context ---\n")

        return {"context": retrieved_texts, "question": query}

    return (
        RunnablePassthrough()  # Handle input properly
        | retrieve_and_format
        | prompt
        | llm
        | StrOutputParser()
    )


# Initialize the RAG chain
rag_chain = create_rag_chain()

# Get user input
question = st.chat_input("Ask your question here...")

if question:
    # Add user question to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # Display user question
    with st.chat_message("user"):
        st.markdown(question)

    # Get chatbot response
    with st.spinner("Thinking..."):
        response = rag_chain.invoke(question)

    # Add chatbot response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(response)


# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.runnable import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from dotenv import load_dotenv
# import streamlit as st

# # Load environment variables
# load_dotenv()

# index_name = "chat-with-data"

# # Initialize Embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Load Vector Store
# vectorstore = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )


# def create_retriever():
#     """Creates a retriever from a vector store."""
#     return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# def create_prompt_template():
#     """Creates a well-structured prompt template for better responses."""
#     template = """
#     You are an AI assistant tasked with answering user questions based on the provided context.
#     Read the context carefully and give a well-explained, structured answer in a friendly and professional tone.

#     If the context does not provide sufficient information, simply say: "I don't have enough information to answer this question."

#     Try to provide a detailed answer while keeping it easy to understand. Structure your answer in paragraphs and use bullet points if needed.

#     Always conclude your response with: "Thanks for asking!"

#     Context:
#     {context}

#     Question: {question}

#     Answer:
#     """

#     return ChatPromptTemplate.from_template(template)


# def create_rag_chain():
#     """Creates a RAG (Retrieval-Augmented Generation) chain with improved processing."""

#     retriever = create_retriever()
#     prompt = create_prompt_template()

#     # Use a better model for detailed responses
#     llm = ChatGoogleGenerativeAI(model="gemini-pro")

#     def retrieve_and_format(query):
#         """Retrieves context and formats it properly before passing it to the LLM."""
#         retrieved_docs = retriever.invoke(query)

#         # Extracting text from retrieved documents
#         retrieved_texts = "\n\n".join(
#             [doc.page_content for doc in retrieved_docs])

#         # Debugging step: print retrieved texts
#         print("\n--- Retrieved Context ---\n",
#               retrieved_texts, "\n--- End Context ---\n")

#         return {"context": retrieved_texts, "question": query}

#     return (
#         RunnablePassthrough()  # Handle input properly
#         | retrieve_and_format
#         | prompt
#         | llm
#         | StrOutputParser()
#     )


# def ask_question(rag_chain, question):
#     """Asks a question using the RAG pipeline and returns the answer."""
#     response = rag_chain.invoke(question)
#     print(f"\nQuestion: {question}")
#     print(f"\nAnswer: {response}\n")
#     return response


# if __name__ == "__main__":
#     rag_chain = create_rag_chain()

#     # Example questions for debugging
#     ask_question(
#         rag_chain, "What is Generative AI and its real-world applications?")


# from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema.runnable import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from dotenv import load_dotenv
# import streamlit as st

# # Load environment variables
# load_dotenv()

# index_name = "chat-with-data"
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# vectorstore = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# # Streamlit app title
# st.title("🤖 RAG Chatbot")
# st.write("Welcome to the RAG Chatbot! Ask me anything about Generative AI.")

# # Initialize session state to store chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])


# def create_retriever():
#     """Creates a retriever from a vector store."""
#     return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# def create_prompt_template():
#     """Creates a prompt template for answering questions."""
#     template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
#     {context}
#     Question: {question}
#     Helpful Answer:"""

#     return ChatPromptTemplate.from_template(template)


# def create_rag_chain():
#     """Creates a RAG (Retrieval-Augmented Generation) chain."""
#     retriever = create_retriever()
#     prompt = create_prompt_template()
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

#     return (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )


# # Initialize the RAG chain
# rag_chain = create_rag_chain()

# # Get user input
# question = st.chat_input("Ask your question here...")

# if question:
#     # Add user question to chat history
#     st.session_state.messages.append({"role": "user", "content": question})

#     # Display user question
#     with st.chat_message("user"):
#         st.markdown(question)

#     # Get chatbot response
#     with st.spinner("Thinking..."):
#         response = rag_chain.invoke(question)

#     # Add chatbot response to chat history
#     st.session_state.messages.append(
#         {"role": "assistant", "content": response})

#     # Display chatbot response
#     with st.chat_message("assistant"):
#         st.markdown(response)
