<h1 style="text-align: center; font-weight: bold; font-size: 45px;">Retriever in RAG:</h1>

## **What is Retriever:**

Retrieval is the process of finding and extracting relevant information from a large collection of data sources or documents. This information is then used by a generative model to produce a more accurate and context-aware response. Retrieval ensures that the generative model bases its responses on the latest or domain-specific data, improving accuracy and relevance.

## **Vector Store as a Retrieval:**

- A vector store is a specialized database that holds vector embeddings—mathematical representations of text, images, or other data. When data is stored as vectors, it enables the system to compare items based on the distance between these vectors, which corresponds to similarity in meaning or context.
- This is particularly useful in RAG for tasks where retrieving based on conceptual similarity (rather than exact keywords) is beneficial. Vector stores are often used together with frameworks like FAISS, Weaviate, or Pinecone for high-speed, scalable retrievals in large datasets.
- The vector store provides a function called semantic search. This function takes your query, goes to the database, and searches for the most relevant vectors related to the query. It then retrieves and returns the information that best answers the query based on semantic similarity, rather than just exact keywords. This allows for a more accurate and contextually meaningful result, even if the wording differs from the stored data.

## **Vector Store Retrieval Method:**

### **1. Similarity Search:**

- Similarity search uses vector representations to retrieve information based on closeness in the embedding space, meaning it retrieves content related in context or meaning, even if the exact words don’t match. This is especially useful in RAG for finding relevant information on less structured queries.
- By ranking possible matches based on how similar they are, similarity search helps RAG systems find and prioritize the most relevant information that closely matches the user’s query. This improves the accuracy of the responses generated. Similarity search is commonly used in recommendation systems and for retrieving content in applications like search engines and virtual assistants.
- If you want to retrieve the most relevant data chunk then you can use the Similarity Search.

### **2. Semantic Search:**

- Semantic search is a method of searching that aims to understand the meaning behind the words in your query, rather than just matching exact words. It seeks to grasp the context, intent, and relationships between words to deliver more relevant and accurate results.
- In Retrieval-Augmented Generation (RAG), semantic search makes sure that the retrieved documents are highly relevant by matching the concept of the query, not just the keywords. This is especially helpful for handling tricky or unclear queries.
- If you want to retrieve multiple similar data chunks then you can use the Semantic Search.

### **3. MMR (Maximal Marginal Relevance):**

- MMR is an approach used to diversify search results. It selects results not only based on relevance but also by ensuring a variety in the retrieved items to avoid redundant information.
- MMR prevents the retrieval of similar data points, balancing between relevance to the query and novelty. This means that when presenting multiple documents or data pieces, the model avoids repetitive or overly similar content, thereby providing a broader perspective or more comprehensive information.
- If you want to ensure a balance between relevancy and diversity in the items that you retrieved, so for this you use Maximal Marginal Relevance.

---

## **Types of LangChain Retrieval:**

- **02_SVMRetriever:** Utilizes Support Vector Machines (SVM) for retrieving relevant documents based on learned decision boundaries, optimizing for high-dimensional data.

- **03_TFIDFRetriever:** Employs the Term Frequency-Inverse Document Frequency (TF-IDF) model to rank documents based on the importance of terms, making it effective for keyword-based searches.

- **04_MultiQueryRetriever:** Allows multiple queries to be processed simultaneously, improving efficiency by returning relevant results for each query in one operation.

- **05_ContextualCompressionRetriever:** Reduces the amount of context in retrieved documents while retaining essential information, optimizing for faster processing and retrieval.

- **06_MultiVectorRetriever:** Supports multiple vector retrievals, enabling the system to pull various embeddings for a comprehensive understanding of the input query.

- **07_BM25Retriever:** Utilizes the BM25 algorithm, a popular ranking function that scores documents based on term frequency and document length, enhancing relevance in information retrieval.

- **08_ParentDocumentRetriever:** trieves documents based on their hierarchical relationship to parent documents, ensuring context is - preserved in the retrieval process.

- **09_SelfQueryRetriever:** Enables the model to query its own retrieved results, enhancing relevance by allowing iterative refinement of search outcomes.

## **Metadata Processing:**


### **3 Things in vector store:**
1. Vector Embedding.
2. Vector Associated text.
3. Vector Associated Metadata.

- When we retrieve the data chunks in our Vector Store, these chunks retrieve in the form of text and it given metadata. It can't retrieve vector embedding just retrieve text and metadata we use vector for just matching purpose. 

