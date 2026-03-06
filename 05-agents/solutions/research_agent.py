import os
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()




# Define tool schemas
class CalculatorInput(BaseModel):
    """Input for calculator."""

    expression: str = Field(description="The mathematical expression to evaluate")


class SearchInput(BaseModel):
    """Input for search."""

    query: str = Field(description="The search query")

called_tools = []

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    called_tools.append("calculator")
    print(f"[tool] calculator called with expression: {expression}")
    
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    called_tools.append("search")
    print(f"[tool] search called with query: {query}")
    
    q = query.lower().strip()

    search_responses = {
    "population of Tokyo": "14,254,039",
    
    "capital of France": "Paris",
}
    
    for key, response in search_responses.items():
        if key in q:
            return f'Search results for "{query}": {response}'

    
    return f'Search results for "{query}": Aucun résultat prédéfini trouvé. Résumé générique: information non spécifique sur \"{query}\".'


def main():
    
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    
    agent = create_agent(model, tools=[calculator, search])

    
    queries = [
        "What is the population of Tokyo multiplied by 2?",
        "Search for the capital of France and tell me how many letters are in its name"       
    ]

    for query in queries:
        called_tools.clear()  # réinitialise pour chaque requête
        print(f" User: {query}")
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        last_message = response["messages"][-1]
        print(f" Agent: {last_message.content}")
        print(f" Tools called for this query: {called_tools}\n")
    
    

if __name__ == "__main__":
    main()

