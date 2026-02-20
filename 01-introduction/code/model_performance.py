import time
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os 
load_dotenv()

question = "Explain the difference between machine learning and deep learning."

models = [
    {"name": "gpt-5", "description": "Most capable"},
    {"name": "gpt-5-mini", "description": "Fast and efficient"},
]

# 4. Create a function to test each model:
#    - Accept model_name as parameter
#    - Create ChatOpenAI instance with that model
#    - Measure start time with time.time()
#    - Invoke the model with the question
#    - Measure end time and calculate duration
#    - Return a dict with name, time, length, and response

def test_model (name):

    model = ChatOpenAI(
            model = name,
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),)
    
    start_time = time.time()
    response = model.invoke(question)
    duration = (time.time() - start_time) * 1000
    
    measures = {"name": name, "time": duration, "length": len(response.content), "response": response.content}

    return measures


# 5. Loop through models list:
#    - Call test_model() for each model
#    - Display results in a formatted table
#    - Use .ljust() for consistent column widths

def main():
    results = []
    for m in models:
        name = m["name"]
        measures = test_model(name)
        print(f"\nResponse from {name}:")
        print(measures["response"])
        quality = input("Rate the quality of the response (1-5): ")
        measures["quality"] = quality
        results.append(measures)

    
    table_header = f"{'Model'.ljust(15)} | {'Time'.ljust(10)} | {'Length'.ljust(10)} | Quality"
    print("\n" + table_header)
    print("-" * len(table_header))
    for r in results:
        print(f"{r['name'].ljust(15)} | {str(round(r['time'],2)).ljust(10)} ch | {str(r['length']).ljust(10)} | {int(r.get('quality','0')) * '⭐'}")

if __name__ == "__main__":    main()
