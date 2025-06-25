# Prompt Template Docs: https://python.langchain.com/v0.2/docs/concepts/#prompt-templateshttps://python.langchain.com/v0.2/docs/concepts/#prompt-templates

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# -------------------------- Simple Prompt Template -----------------------------------------------

# PART 1 (A): Create a Template using ChatPromptTemplate Class.
template: str = "Tell me about {topic} in detail and also tell me the benefits of {topic}."
prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic": "Gen AI"})
print(prompt)


# PART 1 (B): Create a Template using PromptTemplate Class.
prompt_template: PromptTemplate = PromptTemplate(
    template="Tell me about {topic} in detail and also tell me the benefits of {topic}.",
    input_variables=["topic"]
)
prompt = prompt_template.format(topic="Gen AI")
print(prompt)


# -------------------------- Multiple Placeholders Prompt Template -----------------------------------------------

# PART 1 (A): Create a Template using ChatPromptTemplate Class.
template: str = "You are a helpful {topic} Assistant. I want to learn {course}, Create a roadmap for this {course} I have only {time} months & in this time, I cover this course."
prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic": "AI", "course": "Gen AI", "time": "6"})
print(prompt)


# PART 1 (B): Create a Template using PromptTemplate Class.
prompt_template: PromptTemplate = PromptTemplate(
    template="You are a helpful {topic} Assistant. I want to learn {course}, Create a roadmap for this {course} I have only {time} months & in this time, I cover this course.",
    input_variables=["topic", "course", "time"]
)
prompt = prompt_template.format(topic="AI", course="Gen AI", time="6")
print(prompt)


# -------------------------- Prompt with System and Human Messages (Using Tuples) -----------------------------------------------

message = [
    ("system","You are a helpful assistant who can guide user to this {topic}"),
    ("human","What is {content}? & tell me the benefits of {content} & tell me core components of this {content}."),
]

prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(message)
prompt = prompt_template.invoke({"topic": "AI", "content": "Gen AI"})
print(prompt)

#  This does NOT work:
message = [
    ("system","You are a helpful assistant who can guide user to this {topic}"),
    HumanMessage,"What is {content}? & tell me the benefits of {content} & tell me core components of this {content}."
]

prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(message)
prompt = prompt_template.invoke({"topic": "AI", "content": "Gen AI"})
print(prompt)