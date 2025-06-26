"""
# Simple Bot - Integrating LLMs in Graphs

Now we will finally start coding agents.

For this first agent we want to 
- Define a `state` structure with a list of `HumanMessage` objects
- Initialixe an OpenAI model through LangChain
- Building the graph for the Agent
"""

from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv 
import os

load_dotenv()

# a new state for the chat bot
class AgentState(TypedDict):
    messages : List[HumanMessage]   # the state is a list of human messages

# initialize llm
llm = ChatOpenAI(model="gpt-4o")

# define llm node
def process(state: AgentState) -> AgentState:
    """Invokes the llm, which will process the response from our input messages"""    
    
    response = llm.invoke(state['messages'])
    print(f"\nAI: {response.content}")
    
    return state

# construct graph

graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()


# 'chat interface'
user_input = input("Enter: ")

while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")


"""
This llm wrapper has no memory. If we tell him our name and then we ask again, 
he will not know the answer. See how easy it is to implement memory in the next section. 
"""