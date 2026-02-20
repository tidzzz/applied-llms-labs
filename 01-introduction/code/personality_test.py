import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

def main():

    systemMessages = [("Pirate", "You are a pirate. Answer all questions in pirate speak with 'Arrr!' and nautical terms."),("Analyst", "You are a professional business analyst. Give precise, data-driven answers."),("Teacher", "You are a friendly teacher explaining concepts to 8-year-old children.")]

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )
    

    for s in systemMessages:
        messages = [
            SystemMessage(
                content= s[1]
            ),
            HumanMessage(content="What is artificial intelligence?"),
        ]

        response = model.invoke(messages)

        print("Response of " + s[0] + "\n")
        print(response.content)
    print("\n✅ Notice how the SystemMessage influenced the response style!")


if __name__ == "__main__":
    main()
