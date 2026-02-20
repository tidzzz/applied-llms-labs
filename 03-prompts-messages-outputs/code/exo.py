import json
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    
    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),)
    
    examples = [
        {"input": "Gaming laptop with RTX 4070, 32GB RAM, $1,499", "output": {
                                                                    "name": "Gaming laptop",
                                                                    "price": "$1,499",
                                                                    "category": "laptop",
                                                                    "highlight": "RTX 4070, 32GB RAM"
                                                                    }},
        {"input": "Organic cotton t-shirt in blue, comfortable fit, $29.99", "output": {
                                                                    "name": "t-shirt",
                                                                    "price": "$29.99",
                                                                    "category": "Clothes",
                                                                    "highlight": "Organic cotton, comfortable fit"
                                                                    }},
        {"input": "Premium wireless headphones with noise cancellation, $199", "output": {
                                                                    "name": "wireless headphones",
                                                                    "price": "$199",
                                                                    "category": "headphones",
                                                                    "highlight": "noise cancellation"
                                                                    }},
    ]

    example_template = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_template,
        examples=examples,
    )


    final_template = ChatPromptTemplate.from_messages(
        [
            ("system", "convert product descriptions into a specific JSON format based on these examples (Use double quotes for keys and strings.):"),
            few_shot_template,
            ("human", "{input}"),
        ]
    )

    chain = final_template | model


    result = chain.invoke({"input":"Smartphone with camera of 30Mpx and a Snapdragon 835, $400"})
    json_result = json.loads(result.content)
    print(json.dumps(json_result, indent=4))    
    

if __name__ == "__main__":
    main()
