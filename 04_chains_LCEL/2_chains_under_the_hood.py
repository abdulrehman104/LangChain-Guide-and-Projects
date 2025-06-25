# Runnable Docs:         https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable
# RunnableLambda Docs:   https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html
# RunnableSequence Docs: https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableSequence.html

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence

# Load environment variables from .env
load_dotenv()

# Create a ChatGoogleGenerativeAI model
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful {topic} Assistant"),
        ("human", "Tell me What is {contant}? and also tell me the benefits of {contant}")
    ]
)

# Create individual runnables (steps in the chain)

# Step 1: Format the input (this formats the prompt for the language model)
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

# Step 2: Invoke the model (this sends the formatted prompt to the LLM)
invoke_model = RunnableLambda(lambda x: model.invoke(x))

# Step 3: Parse the output (this extracts the text output from the LLM)
parse_output = RunnableLambda(lambda x: x.content)

# Chain these steps together into a sequence
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic":"AI","contant":"LangChain & LangGraph"})

# output
print(response)



# ------------------------------------------------ Example Of RunnableLambda ------------------------------------------------

# RunnableLambda: (Agr chain ko invoke krna he to ham Runnable bnaye ge )
# What it is: RunnableLambda is a wrapper around any Python function or callable. It allows you to take a simple Python function (a callable) and make it part of a "runnable" chain that can be executed in a step-by-step manner, like chaining different tasks together.

# Why it's useful: You can turn any simple function into a "runnable" and integrate it into complex workflows or chains. For instance, you might want to perform some operation in between model calls, like formatting the input or parsing the output, and RunnableLambda helps you insert that step easily into a processing chain.


# Define a simple function that multiplies a number by 2
def multiply_by_2(x):
    return x * 2


# Wrap the function in RunnableLambda
multiply_chain = RunnableLambda(multiply_by_2)

# Now, you can run it
result = multiply_chain.invoke(5)
print(result)  # Output: 10



# ------------------------------------------------ Example Of Runnable Suquence ------------------------------------------------

# RunnableSequence: (Ye sare runnable ko sequence mn run krta he)
# What it is: RunnableSequence is a way to chain together multiple "runnables" (which can include RunnableLambda or other runnables like language model calls) in a sequence. Each step in the sequence processes the output of the previous step. It essentially creates a pipeline where the result of one step is fed into the next step.

# Why it's useful: RunnableSequence is helpful when you have multiple tasks to perform in sequence, where each task depends on the output of the previous one. It allows you to set up a workflow of different processing steps, like formatting, calling an LLM, and parsing output, all in a neat chain.

# Lambda Function Syntax:   lambda arguments: expression

# Step 1: Format the input (this formats the prompt for the language model)
format_prompt = RunnableLambda(
    lambda x: f"Tell me {x['joke_count']} jokes about {x['topic']}.")

# Step 2: Invoke the model (this sends the formatted prompt to the LLM)
invoke_model = RunnableLambda(lambda x: model.invoke(x))

# Step 3: Parse the output (this extracts the text output from the LLM)
parse_output = RunnableLambda(lambda x: x.content)

# Chain these steps together into a sequence
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Execute the chain
response = chain.invoke({"topic": "AI", "joke_count": 3})
print(response)