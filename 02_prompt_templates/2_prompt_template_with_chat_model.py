from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# Create a ChatGoogleGenerativeAI model
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash")


# -------------------------- Zero Shot Prompt Template With ChatModel Response -----------------------------------------------

template: str = "What is {contant}? and tell me the benefits of {contant}?"
prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(template)

# Get response to the LLM model
prompt = prompt_template.invoke({"contant": "Gen AI"})
response = model.invoke(prompt)
print(response.content)


# -------------------------- Multiple Placeholders Prompt Template With ChatModel Response -----------------------------------------------

template: str = "You are a helpful {topic} Assistant. I want to learn {course}, Create a roadmap for this {course} I have only {time} months & in this time, I cover this course."
prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(template)

# Get response to the LLM model
prompt = prompt_template.invoke({"topic": "AI", "course": "Gen AI", "time": 6})
response = model.invoke(prompt)
print(response.content)


# -------------------------- Few Shot Prompt Template  with System and Human Messages (Using Tuples) -----------------------------------------------

message = [
    ("system","You are a helpful assistant who can guide user to this {topic}"),
    ("human","What is {content}? & tell me the benefits of {content} & tell me core components of this {content}."),
]

prompt_template = ChatPromptTemplate.from_messages(message)
prompt = prompt_template.invoke({"topic": "AI", "content": "Gen AI"})
response = model.invoke(prompt)
print(response.content)