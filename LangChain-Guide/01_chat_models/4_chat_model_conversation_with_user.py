from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatGoogleGenerativeAI model
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Use a list to store messages
chat_history: list = []

# Set an initial system message (optional)
message = SystemMessage(content="You are a helpful assistant")
chat_history.append(message)  # Add system message to chat history

# Chat loop
while True:
    query: str = input("Enter your query: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    # Get AI response using history
    response = model.invoke(query)
    chat_history.append(AIMessage(content=response.content))
    print(f"AI Message:", response.content)

print("Chat History")
print(chat_history)