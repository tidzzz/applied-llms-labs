import os
from typing import Literal, Optional
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field



load_dotenv()


class WeatherInput(BaseModel):

    city: str = Field(description="City name")
    units: Optional[Literal["celsius", "fahrenheit"]] = Field(
        default="fahrenheit", description="Units for temperature (celsius or fahrenheit)"
    )


@tool(args_schema=WeatherInput)
def get_weather(city: str, units: str = "fahrenheit") -> str:
    """Get current weather for a city."""
    # Simulate API call
    temps = {"Seattle": 62, "Paris": 18, "Tokyo": 24, "London": 14, "Sydney": 26}
    temp = temps.get(city, 72)
    if units == "celsius":
        temp = (temp - 32) * 5/9
        return f"Current temperature in {city}: {temp}°C, partly cloudy"
    return f"Current temperature in {city}: {temp}°F, partly cloudy"

def main():


    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    model_with_tools = model.bind_tools([get_weather])

    response1 = model_with_tools.invoke([HumanMessage(content="What's the weather in Seattle, in celsius?")])

    tool_call = response1.tool_calls[0]
    tool_result = get_weather.invoke(tool_call["args"])

    messages = [
        HumanMessage(content="What's the weather in Seattle?"),
        AIMessage(
            content=str(response1.content),
            tool_calls=response1.tool_calls,
        ),
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"],
        ),
    ]

    final_response = model.invoke(messages)
    print(f"AI: {final_response.content}\n")

if __name__ == "__main__":    main()