import os
import asyncio
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google.adk.tools.tool_context import ToolContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import google_search
from google.adk.tools import agent_tool 
from google.adk.planners import PlanReActPlanner

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

print("Libraries imported.")

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
os.environ["GOOGLE_API_KEY"] = "AIzaSyCRqwQNdLTpdEScgqNYdm7NOxUGUs7UhLc"

print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")

MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"
#MODEL_GEMINI_2_0_FLASH = "gemini-1.5-pro"
print("Environment configured.")

# #title Define Tools

# @title Define the get_weather Tool
def get_weather(city: str, tool_context: ToolContext) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city(e.g., "New York", "London", "Tokyo")

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error')
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    # --- Read preference from state ---
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")

    city_normalized = city.lower().replace(" ", "")

    # Mock weather data (always stored in Celsius internally)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]

        # Format temperature based on state preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9/5) + 32 # Calculate Fahrenheit
            temp_unit = "°F"
        else: # Default to Celsius
            temp_value = temp_c
            temp_unit = "°C"

        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        result = {"status": "success", "report": report}
        print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")

        # Example of writing back to state (optional for this tool)
        tool_context.state["last_city_checked_stateful"] = city
        print(f"--- Tool: Updated state 'last_city_checked_stateful': {city} ---")

        return result
    else:
        # Handle city not found
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")
        return {"status": "error", "error_message": error_msg}

# @title Define the say_helo Tool
def say_hello(name: Optional[str] = None) -> str:
   """Provides a simple greeting. If a name is provided, it will be used.
   Args:
        name (str, optional): The name of the person to greet. Defaults to a generic greeting if not provided.
   Returns:
        str: A friendly greeting message.
   """
   if name:
      print(f"--- Tool: say_hello called with name: {name} ---")
      greeting = f"Hello, {name}!"
   else:
      print(f"--- Tool: say_hello called without a specific name (name_arg_value: {name}) ---")
      greeting = "Hello there!"
   return greeting

def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation."""
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."

print("Greeting and Farewell tools defined.")

# @title Define Greeting and Farewell Sub-Agents
# --- Greeting Agent ---
greeting_agent = None
try:
    greeting_agent = Agent(
        model = MODEL_GEMINI_2_0_FLASH,
        name="greeting_agent",
        instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
                    "Use the 'say_hello' tool to generate the greeting. "
                    "If the user provides their name, make sure to pass it to the tool. "
                    "Do not engage in any other conversation or tasks.",
        description="Handles simple greetings and hellos using the 'say_hello' tool.", # Crucial for delegation
        tools=[say_hello],
    )
    print(f"Agent '{greeting_agent.name}' created using model '{greeting_agent.model}'.")
except Exception as e:
    print(f"Could not create Greeting agent. Check API Key ({greeting_agent.model}). Error: {e}")

# --- Farewell Agent ---
farewell_agent = None
try:
    farewell_agent = Agent(
        model = MODEL_GEMINI_2_0_FLASH,
        name="farewell_agent",
        instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
                    "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation "
                    "(e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
                    "Do not perform any other actions.",
        description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.", # Crucial for delegation
        tools=[say_goodbye],
    )
    print(f"Agent '{farewell_agent.name}' created using model '{farewell_agent.model}'.")
except Exception as e:
    print(f"Could not create Farewell agent. Check API Key ({farewell_agent.model}). Error: {e}")

# --- Google Search ---
google_search_agent = None
try:
    google_search_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,
        name="google_search_agent",
        instruction="You are an expert researcher. You always stick to the facts",
        description="Agent to answer questions using Google search",
        tools=[google_search],
        planner=PlanReActPlanner()
    )
    print(f"Agent '{google_search_agent.name}' created using model '{google_search_agent.model}'.")
except Exception as e:
    print(f"Could not create Google search agent. Check API Key ({google_search_agent.model}). Error: {e}")

# @title Define the before_model_callback Guardrail
def block_model_guardrail(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """
    Inspects the latest user message for 'BLOCK'. If found, blocks the LLM call and returns a predefined LlmResponse.
    Otherwise, returns None.
    """
    agent_name = callback_context.agent_name
    print(f"--- Callback: block_keyword_guardrail running for agent: {agent_name} ---")
    # Extract the text from the latest user message in the request history
    last_user_message_text = ""
    if llm_request.contents:
        # Find the most recent message with role 'user'
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts:
                # Assuming text is in the first part for simplicity
                if content.parts[0].text:
                    last_user_message_text = content.parts[0].text
                    break # Found the last user message text

    print(f"--- Callback: Inspecting last user message: '{last_user_message_text[:100]}...' ---") # Log first 100 chars

    # --- Guardrail Logic ---
    keyword_to_block = "BLOCK"
    if keyword_to_block in last_user_message_text.upper(): # Case-insensitive check
        print(f"--- Callback: Found '{keyword_to_block}'. Blocking LLM call! ---")
        # Optionally, set a flag in state to record the block event
        callback_context.state["guardrail_block_keyword_triggered"] = True
        print(f"--- Callback: Set state 'guardrail_block_keyword_triggered': True ---")

        # Construct and return an LlmResponse to stop the flow and send this back instead
        return LlmResponse(
            content=types.Content(
                role="model", # Mimic a response from the agent's perspective
                parts=[types.Part(text=f"I cannot process this request because it contains the blocked keyword '{keyword_to_block}'.")],
            )
            # Note: You could also set an error_message field here if needed
        )
    else:
        # Keyword not found, allow the request to proceed to the LLM
        print(f"--- Callback: Keyword not found. Allowing LLM call for {agent_name}. ---")
        return None # Returning None signals ADK to continue normally

# @titel Define the Weather Agent
# Use one of model constants defined earlier
AGENT_NAME = "weather_agent_v1"

weather_agent_team = Agent(
        name=AGENT_NAME,
        model=MODEL_GEMINI_2_0_FLASH,
        description="The main coordinator agent. Handles weather requests or other domain requests and delegates greetings/farewells to specialists.",
        global_instruction="Please use Chinese to answer all questions.",
        instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
                    "You have two tools:"
                    "1. 'get_weather': Use this tool ONLY for specific weather requests (e.g., 'weather in London'). "
                    "2. 'google_search_agent': If the requested city is not found, I will perform a Google search to find the weather information useing the this tool"
                    "You have specialized sub-agents: "
                    "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
                    "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
                    "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'."
                    "If it's a weather request, handle it yourself using 'get_weather'. "
                    "If it's a other reqeust, handle it usging google search agnet. '"
                    "For anything else, respond appropriately or state you cannot handle it.",
        tools=[get_weather,
               agent_tool.AgentTool(agent=google_search_agent)],
        sub_agents=[greeting_agent, farewell_agent]
        #output_key="last_weather_report",
        #before_model_callback=block_model_guardrail
        )

print(f"Agent '{weather_agent_team.name}' created using model '{MODEL_GEMINI_2_0_FLASH}'.")

# @title Setup Session Service and Runner

# --- Session Management ---
# Define constants for identifying the interaction context
APP_NAME = "weather_tutorial_app"
USER_ID = "user_1"
SESSION_ID = "session_001"

# Key concept: SessionService stores conversation history & state.
# InMemorySessionService is simple, non-persistent storage for this tutorial.
session_service = InMemorySessionService()
session = None
runner = None

# Define initial state data - user prefers Celsius initially
initial_state = {
    "user_preference_temperature_unit": "Celsius"
}

async def create_session():
    global session
    global initial_state
    
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state)

    print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

# --- Runner ---
# Key Concept: Runner orchestrates the agent execution loop.

async def create_runner():
    global runner 

    runner = Runner(
        agent=weather_agent_team,
        app_name=APP_NAME,
        session_service=session_service)

    print(f"Runner created for agent '{runner.agent.name}'.")

# @titel Define Agent Interaction Function

async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f">>> User Query: {query}")

    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")
        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            break

    print(f"<<< Agent Response: {final_response_text}")

# @title Run the Initial Conversation
async def run_conversation():
    """
    await call_agent_async(query = "Hello there!",
                           runner=runner,
                           user_id=USER_ID,
                           session_id=SESSION_ID)

    await call_agent_async("What is the weather like in London?",
                                       runner=runner,
                                       user_id=USER_ID,
                                       session_id=SESSION_ID)

    stored_session = session_service.sessions[APP_NAME][USER_ID][SESSION_ID]
    stored_session.state["user_preference_temperature_unit"] = "Fahrenheit"

    await call_agent_async(query= "This is a BLOCK test request.",
                               runner=runner,
                               user_id=USER_ID,
                               session_id=SESSION_ID
                              )
    await call_agent_async(query= "Tell me the weather in New York.",
                               runner=runner,
                               user_id=USER_ID,
                               session_id=SESSION_ID
                              )
    await call_agent_async(query= "Tell me the weather in Beijing today。",
                               runner=runner,
                               user_id=USER_ID,
                               session_id=SESSION_ID
                              )

    """
    await call_agent_async(query= "帮我制定一个今年国庆假期在山东的游玩计划。",
                               runner=runner,
                               user_id=USER_ID,
                               session_id=SESSION_ID
                              )
    """
    await call_agent_async(query = "Thanks, bye!",
                               runner=runner,
                               user_id=USER_ID,
                               session_id=SESSION_ID)
    """

if __name__ == "__main__":
    asyncio.run(create_session())
    asyncio.run(create_runner())
    asyncio.run(run_conversation())
