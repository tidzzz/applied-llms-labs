# 1. Import required modules
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

# 2. Load environment variables
load_dotenv()

# 3. Create the ChatOpenAI model
model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

# 4. Initialize conversation history list with a SystemMessage for personality
messages = [
    SystemMessage(content="You are a helpful assistant.")
]


def main():
    print("🤖 Welcome to the Chatbot! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        messages.append(HumanMessage(content=user_input))
        response = model.invoke(messages)
        messages.append(AIMessage(content=response.content))

        print("Bot:", response.content)

    print(f"\n✅ Conversation ended. Total messages exchanged: {len(messages)}")

if __name__ == "__main__":    main()

