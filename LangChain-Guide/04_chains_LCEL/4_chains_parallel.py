# RunnableParallel Docs: https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableParallel.html

# Parallel Chains:
# A parallel chain is a structure that allows you to execute multiple tasks or functions simultaneously, rather than sequentially. This can be particularly useful when you want to run independent operations that don’t depend on each other's output and can be processed concurrently, reducing the overall time needed for execution.

# RunnableParallel: (agr ham chate hn ke hamare pipline parallel run ho to ham isko use kre ge and but isko jab use kre ge jab chain ke output and input independent ho)
# RunnableParallel is a construct in LangChain that allows you to run multiple chains (or tasks) in parallel, meaning they execute at the same time, rather than sequentially. It is useful when you want to perform several operations independently of each other, and you want to combine their results later.

# This code uses LangChain to create a pipeline that generates a product review, focusing on listing both the pros and cons of a product. The LangChain Expression Language (LCEL) is used to simplify this multi-step process.

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)


# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human","Given these features: {features}, list the pros of these features."),
        ]
    )
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list the cons of these features."),
        ]
    )
    return cons_template.format_prompt(features=features)


# Combine pros and cons into a final review
def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


# Simplify branches with LCEL
pros_branch_chain = RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()


cons_branch_chain = RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()


# Create the combined chain using LangChain Expression Language (LCEL)
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

# Run the chain
result = chain.invoke({"product_name": "Microsoft Surface Laptop"})

# Output
print(result)