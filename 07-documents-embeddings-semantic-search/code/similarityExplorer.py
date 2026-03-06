import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def get_embeddings_endpoint():
    """Get the Azure OpenAI endpoint, removing /openai/v1 suffix if present."""
    endpoint = os.getenv("AI_ENDPOINT", "")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint.replace("/openai/v1", "")
    elif endpoint.endswith("/openai/v1/"):
        endpoint = endpoint.replace("/openai/v1/", "")
    return endpoint

docs = [
    Document(page_content="Machine learning models can recognize patterns in data"),
    Document(page_content="The recipe calls for flour, eggs, and butter"),
    Document(page_content="Python is a popular programming language for AI"),
    Document(page_content="The sunset painted the sky in shades of orange"),
    Document(page_content="The stock market can be volatile and unpredictable"),
    Document(page_content="The cat sat on the windowsill, basking in the sun"),
    Document(page_content="The novel explores themes of love and loss"),
    Document(page_content="The conference will cover the latest advancements in AI"),
]




queries = [
    "How does AI learn?",
    "What ingredients do I need for baking?",
    "Beautiful evening colors",
]



def main():
    
    
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )
    
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
    
    print("  Vector Store and Semantic Search\n")
    for query in queries:
        print(f'\nQuery: "{query}"\n')
        results = vector_store.similarity_search_with_score(query, k=3)
        for doc, score in results:
            print(f"Score: {score:.4f} - {doc.page_content}")

if __name__ == "__main__":    
    
    main()