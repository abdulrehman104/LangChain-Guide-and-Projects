# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Initialize an instance of the ChatGoogleGenerativeAI with specific parameters
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)


# Invoke the model with a message
result = model.invoke( "Hey I want to learn problem-solving skills how can I learn it. ")

print("Full result:")
print(result)

print("Content only:")
print(result.content)