# Document Loader DOCS = https://python.langchain.com/api_reference/community/document_loaders.html

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader

# ============================ PyPDFLoader ================================================
pdf_loader = PyPDFLoader(
    "C:/Users/rehma/AppData/Local/Microsoft/Windows/INetCache/IE/N8JW5PR9/What_Are_AI_Agents[1].pdf")
pdf_data = pdf_loader.load()

print(len(pdf_data))
print(pdf_data)


print(pdf_data[35].page_content)
print(pdf_data[35].metadata['source'])
print(pdf_data[35].metadata['page'])


# ============================ TextLoader ================================================

text_loader = TextLoader("/content/linkedin.txt")
text_data = text_loader.load()

print(text_data)

print(text_data[0].metadata["source"])
print(text_data[0].page_content[0:500])

# ============================ CSVLoader ================================================
csv_loader = CSVLoader("/content/annual-enterprise.csv")
csv_data = csv_loader.load()

print(csv_data)

print(csv_data[1].metadata["source"])
print(csv_data[1].metadata["row"])
print(csv_data[1].page_content)

print(csv_data[2].page_content)
print(csv_data[3].page_content)
print(csv_data[4].page_content)
print(csv_data[5].page_content)
print(csv_data[6].page_content)
print(csv_data[7].page_content)
print(csv_data[8].page_content)
print(csv_data[9].page_content)

# ============================ WebBaseLoader ================================================
web_page_loader = WebBaseLoader("https://www.xevensolutions.com/")
web_page_data = web_page_loader.load()

print(web_page_data)

print(web_page_data[0].metadata["source"])
print(web_page_data[0].metadata["title"])
print(web_page_data[0].metadata["description"])
print(web_page_data[0].page_content[400:500])