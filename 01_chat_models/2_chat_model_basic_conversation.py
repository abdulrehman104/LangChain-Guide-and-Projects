from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# Load environment variables from .env
load_dotenv()

# Initialize an instance of the ChatGoogleGenerativeAI with specific parameters
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# SystemMessage: Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.
# HumanMessagse: Message from a human to the AI model.
# AIMessage: Message from an AI.

message = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Who is Sultan Murad 4 tell me in short")
]

# Invoke the model with messages
response = model.invoke(message)
print(f"Answer from AI: {response.content}")


message = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Who is Sultan Murad 4 tell me in short"),
    AIMessage(content="Sultan Murad IV (1612-1640) was the 16th Ottoman Sultan. He was known for his **brutality and ruthlessness**, but also for **reforming the Ottoman military and regaining lost territories**. He was a **strong ruler** who restored order to the empire after a period of decline."),
    HumanMessage(content="Sultan Murad kitne time tk sultan rhe")
]

# Invoke the model with message
response = model.invoke(message)
print(f"Answer from AI: {response.content}")