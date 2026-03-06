from langchain.agents import create_agent
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

def get_embeddings_endpoint():
    """Get the Azure OpenAI endpoint, removing /openai/v1 suffix if present."""
    endpoint = os.getenv("AI_ENDPOINT", "")
    if endpoint.endswith("/openai/v1"):
        endpoint = endpoint.replace("/openai/v1", "")
    elif endpoint.endswith("/openai/v1/"):
        endpoint = endpoint.replace("/openai/v1/", "")
    return endpoint

def main():
    print(" Agentic RAG System Example\n")

    # 1. Setup embeddings and model
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=get_embeddings_endpoint(),
        api_key=os.getenv("AI_API_KEY"),
        model=os.getenv("AI_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_version="2024-02-01",
    )

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    docs = [
        Document(
            page_content="La nutrition est la science qui étudie les nutriments et leur impact sur la santé. Une bonne nutrition prévient les maladies chroniques et améliore la qualité de vie. Les macronutriments incluent les protéines, les lipides et les glucides.",
            metadata={
                "title": "Bases de la nutrition",
                "source": "nutrition-basics",
                "topic": "introduction",
            },
        ),
        Document(
            page_content="Les protéines sont essentielles pour la construction musculaire et la réparation tissulaire. Les sources principales incluent la viande, le poisson, les œufs, les légumineuses et les produits laitiers. Les besoins quotidiens varient selon l'âge et l'activité physique.",
            metadata={
                "title": "Guide des protéines",
                "source": "proteine-guide",
                "topic": "macronutrients",
            },
        ),
        Document(
            page_content="Les vitamines et minéraux jouent un rôle crucial dans le fonctionnement du corps. La vitamine C renforce l'immunité, le calcium fortifie les os, et le fer prévient l'anémie. Une alimentation équilibrée couvre généralement tous les besoins nutritionnels.",
            metadata={
                "title": "Vitamines et minéraux essentiels",
                "source": "vitamines-mineraux",
                "topic": "micronutrients",
            },
        ),
        Document(
            page_content="La nutrition personnalisée tient compte des antécédents médicaux, des allergies et des objectifs de santé. Les nutritionnistes recommandent une assiette équilibrée avec 50% de fruits et légumes, 25% de protéines et 25% de féculents complets.",
            metadata={
                "title": "Nutrition personnalisée",
                "source": "nutrition-personalisee",
                "topic": "planning",
            },
        ),
    ]
    
    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    @tool
    def search_my_notes(query: str) -> str:
        """Search my personal knowledge base for information."""
        results = vector_store.similarity_search(query, k=3)
        return "\n\n".join(
            f"[{doc.metadata['title']}]: {doc.page_content}"
            for doc in results
        )
    
    agent = create_agent(
        model,
        tools=[search_my_notes],
        system_prompt="You are a helpful assistant that answers questions about nutrition based on the user's personal knowledge base. Use the search tool to find relevant information when needed."
    )

    response = agent.invoke({
    "messages": [HumanMessage(content="Quels sont les principaux nutriments et leurs rôles dans la santé ?")],})
    # Vérifie si l'agent a décidé d'appeler un tool
    tool_calls = []
    for msg in response["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)

    if tool_calls:
        tool_names = [call.get("name", "unknown_tool") for call in tool_calls]
        print(f" Décision agent: utiliser le tool ({', '.join(tool_names)})")
    else:
        print(" Décision agent: ne pas utiliser de tool")
    final_message = response["messages"][-1]
    print(" Answer:", final_message.content)

    
    
if __name__ == "__main__":    
    main()