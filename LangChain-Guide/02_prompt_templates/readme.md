# **Prompt Template in LangChain:**

## **What is Prompt:**

A prompt for a language model is a set of instructions or input provided by a user to guide the model's response, helping it understand the context and generate relevant and coherent language-based output, such as answering questions, completing sentences, or engaging in a conversation.

## **Types of Prompts:**

### **Zero Shot Prompt:**

```PY
# zero shot prompting
# When the task is straightforward or commonly understood (e.g., simple translation, factual answers).
# When you want the model to be more flexible in its response, without being influenced by specific examples.

prompt = "Tell me about yourself."
prompt="can you tell me total number of country in aisa? can you give me top 10 contry name?"
```

### **Few Shot Prompt:**

```py
# Few Short Prompts
# When you need to guide the model to respond in a specific format or with a particular type of answer.
# For tasks that may be more complex and benefit from seeing a couple of examples to better understand the context.

student_description = "AbdulRehman is a student of Gen AI at Piaic. He score a 8.5 GPA. It is known for his programming skills and is an active member of the college's AI Club. He hopes to pursue a career in artificial intelligence after graduating."
prompt = f'''
Please extract the following information from the given text and return it as a JSON object:

name
college
grades
club

This is the body of text to extract the information from:
{student_description}
'''
# A simple prompt to extract information from "student_description" in a JSON format.

```

## **What is Prompt Template:**

Prompt templates in LangChain are predefined structures that help generate prompts for language models. They ensure that the prompts are consistent and tailored to specific tasks, which can improve the quality of the AI's responses.

### **Key Features of ChatPromptTemplate**

1. **Template Creation:**
   You can create a template that includes placeholders for dynamic content. This allows you to define a structure for your prompts, specifying where user inputs and AI responses should go.

2. **Dynamic Formatting:**
   At runtime, you can fill in the placeholders with actual values. This means you can generate prompts on-the-fly based on the current context of the conversation.

3. **Style and Tone Customization:**
   You can specify the style and tone for the AI's responses. For example, you might want the AI to respond in a calm and respectful tone or even in a specific dialect or language.

4. **Message Formatting:**
   The template can format messages according to the specified style and content. This ensures that the AI's responses are coherent and contextually appropriate.

## **Prompt Template vs ChatPrompTemplate:**

In LangChain, **PromptTemplate** and **ChatPromptTemplate** are both used to structure prompts, but they are designed for different contexts:

1. **PromptTemplate**:

   - Best for **single-turn, text-only prompts** where the response is expected in a straightforward text format.
   - Suitable for simpler tasks where you don’t need multi-turn dialogue or conversational context.

2. **ChatPromptTemplate**:
   - Specifically designed for **multi-turn, chat-based prompts** and structured to handle a sequence of messages (e.g., alternating between user and AI responses).
   - Ideal for use with **chat-based language models** (like ChatGPT) and supports a more conversational structure.

In short, use **PromptTemplate** for single-message prompts, and **ChatPromptTemplate** when you need a back-and-forth, chat-like setup.

---
