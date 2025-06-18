from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, RecursiveJsonSplitter, Language
from langchain_text_splitters import HTMLHeaderTextSplitter, HTMLSectionSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter
import requests

# =================================================================== Text Splitter  ===================================================================
# ======================== CharacterTextSplitter ========================

# Define the text to be split into chunks
text = "AI is one of the most exciting advancements in recent years."

# Set the chunk size and overlap
chunk_size = 50  # Maximum size of each chunk
chunk_overlap = 20  # Number of overlapping characters between consecutive chunks

# ============================== CharacterTextSplitter ==============================

# Initialize CharacterTextSplitter with chunk size, overlap, and separator
character_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator="\n\n",  # Splitting the text based on double newlines
    # separator="": Optional - If you want no separator (i.e., split anywhere), but here we use:
)

# Split the text into chunks using the specified settings
c_chunks = character_splitter.split_text(text)

# Print the chunks and their count
print("========== CharacterTextSplitter ==========")
print(c_chunks)  # Outputs the list of chunks generated
print(len(c_chunks))  # Outputs the number of chunks

# ======================== RecursiveCharacterTextSplitter ========================

# Initialize RecursiveCharacterTextSplitter with a list of separators
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ".", ",", ""]
    # The model will try to split the text at the largest separator first, then move down the list
)

# Split the text recursively using the defined separators
r_chunks = recursive_splitter.split_text(text)

# Print the chunks and their count
print("========== RecursiveCharacterTextSplitter ==========")
print(r_chunks)  # Outputs the list of chunks generated
print(len(r_chunks))  # Outputs the number of chunks

# ================================================ Json Splitter  ================================================

# This is a large nested json object and will be loaded as a python dict
json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

json_splitter = RecursiveJsonSplitter(max_chunk_size=500)

# This is a large nested json object and will be loaded as a python dict
json_data = requests.get("https://api.smith.langchain.com/openapi.json").json()

json_splitter = RecursiveJsonSplitter(max_chunk_size=300)

# Recursively split json data - If you need to access/manipulate the smaller json chunks
j_chunks = json_splitter.split_json(json_data)

print("========== RecursiveJsonSplitter ==========")
# print(j_chunks)  # This print huge amount of JSON data
print(len(j_chunks))
print(j_chunks[0])
print(j_chunks[0]["openapi"])
print(j_chunks[0]["info"])
print(j_chunks[0]["paths"])

# ================================================ Code Splitter  ================================================

# ==================== RecursiveCharacterTextSplitter with out language class ====================
# Sample code to be split
code_text = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

class Calculator:
    def __init__(self):
        self.result = 0

    def calculate(self, func, a, b):
        self.result = func(a, b)
        return self.result
"""

# Initialize the RecursiveCharacterTextSplitter with specific chunk sizes
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # Size of each chunk in characters
    chunk_overlap=10,  # Overlap between chunks
    separators=["\n\n", "\n", " "]  # Separators to prioritize splitting
)

# Split the code into chunks
c_chunks = splitter.split_text(code_text)

print("========== RecursiveCharacterTextSplitter ==========")
print(len(c_chunks))
print(c_chunks)

# ==================== RecursiveCharacterTextSplitter with language class ====================

# Sample code to be split
python_code = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""

# Initialize the RecursiveCharacterTextSplitter with specific chunk sizes
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=0
)

# Split the code into chunks
c_l_chunks = python_splitter.split_text(python_code)

print("========== RecursiveCharacterTextSplitter with language ==========")
print(len(c_l_chunks))
print(c_l_chunks)

# ================================================ HTML Splitter  ================================================

# ==================== HTMLHeaderTextSplitter ====================

# Define an HTML string to split based on headers
html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>
        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""

# Specify the headers to split on (h1, h2, h3)
headers_to_split_on = [
    ("h1", "Header 1"),  # Split on <h1> tags, and label as "Header 1"
    ("h2", "Header 2"),  # Split on <h2> tags, and label as "Header 2"
    ("h3", "Header 3"),  # Split on <h3> tags, and label as "Header 3"
]

# Initialize the HTMLHeaderTextSplitter with the specified headers
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on)

# Split the HTML string based on the header tags
header_splits = html_splitter.split_text(html_string)

# Print the split results
print("========== HTMLHeaderTextSplitter ==========")
print(header_splits)  # Output all the header splits
print(header_splits[0].page_content)  # Content of the first split section
print(header_splits[0].metadata)  # Metadata of the first split section

print(header_splits[1].page_content)  # Content of the second split section
print(header_splits[1].metadata)  # Metadata of the second split section
# Accessing specific header metadata
print(header_splits[1].metadata["Header 1"])

# Print metadata for a deeper split section
print(header_splits[4].metadata)
# Access "Header 1" metadata in this split
print(header_splits[4].metadata["Header 1"])
# Access "Header 2" metadata in this split
print(header_splits[4].metadata["Header 2"])
# Access "Header 3" metadata in this split
print(header_splits[4].metadata["Header 3"])
print(header_splits[4].page_content)  # Content of the split section

# ==================== HTMLSectionSplitter ====================

# Define the same HTML string as above for splitting
html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>
        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""

# Initialize HTMLSectionSplitter with the same headers
html_splitter = HTMLSectionSplitter(headers_to_split_on)

# Split the HTML string into sections based on headers
html_header_splits = html_splitter.split_text(html_string)

# Print the split results
print("========== HTMLSectionSplitter ==========")
print(html_header_splits)  # Output all the section splits

print(html_header_splits[0].page_content)  # Content of the first section
print(html_header_splits[0].metadata)  # Metadata of the first section

print(html_header_splits[1].page_content)  # Content of the second section
print(html_header_splits[1].metadata)  # Metadata of the second section
# Access specific "Header 2" metadata
print(html_header_splits[1].metadata["Header 2"])

# ================================================ Markdown Splitter  ================================================

# Define a sample markdown text to split based on headers
markdown_text = """
# Introduction
This is the introduction to the document.

## History
The history section covers the early days of technology.

### Ancient Tech
People have been using technology since the beginning of time.

## Current Technology
We are living in a highly advanced technological world.

### AI and Machine Learning
AI is one of the most exciting advancements in recent years.

# Conclusion
This is the conclusion of the document.
"""

# Specify the markdown headers to split on
headers_to_split_on = [
    ("#", "Header 1"),   # Split on top-level headers ("#")
    ("##", "Header 2"),  # Split on second-level headers ("##")
    ("###", "Header 3"),  # Split on third-level headers ("###")
]

# Initialize the MarkdownHeaderTextSplitter with the specified headers
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# Split the markdown text based on headers
md_header_splits = markdown_splitter.split_text(markdown_text)

# Print the split results
print("========== MarkdownHeaderTextSplitter ==========")
print(md_header_splits)  # Output all the header splits

print(md_header_splits[0].metadata)  # Metadata of the first section
print(md_header_splits[0].page_content)  # Content of the first section

# Print metadata and content for deeper sections
print(md_header_splits[4].metadata)
print(md_header_splits[4].metadata["Header 1"])  # Access "Header 1" metadata
print(md_header_splits[4].metadata["Header 2"])  # Access "Header 2" metadata
print(md_header_splits[4].metadata["Header 3"])  # Access "Header 3" metadata
print(md_header_splits[4].page_content)  # Content of the split section