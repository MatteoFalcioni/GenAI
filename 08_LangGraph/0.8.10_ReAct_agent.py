from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# State with Annotated (standard procedure in LangGraph)
class AgentState(TypedDict):
    """We will have a sequence of messages (either AI-, Human- or Tool- like)"""
    messages: Annotated[Sequence[BaseMessage], add_messages]    # BaseMessage is the parent class for all LangChain messages


# Tools definitions
@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]


# llm + bind tools
model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)


# define model call node
def model_call(state : AgentState) -> AgentState:
    """Call the llm to respond to a given request."""
    # system message
    system_prompt = SystemMessage(content=
                                  "You are an helpful AI assistant, please answer my query to the best of your ability.")
    
    response = model.invoke([system_prompt] + state['messages'])

    # another (more compact way) to update the state, thanks to add_messages:
    return {"messages": [response]}


# define router node
def should_continue(state: AgentState):
    """Router node to decide wether we need to use tools or not"""
    messages = state['messages']
    last_message = messages[-1] # check if the last message is a tool call

    if not last_message.tool_calls:
        return "end"    # go to END
    else:
        return "continue"   # call tool
    
# define tool node
tool_node = ToolNode(tools=tools)
    

## Construct graph 
graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges(source="our_agent",
                            path=should_continue,
                            path_map={
                                "continue" : "tools",
                                "end" : END
                            })

graph.add_edge("tools", "our_agent")    # to complete the tool loop 
app = graph.compile()


# helper function to pretty print
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))


"""
matteo@pcmatteo:~/Learning_AI/GenAI/08_LangGraph$ python 0.8.10_ReAct_agent.py 
================================ Human Message =================================

Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.
================================== Ai Message ==================================
Tool Calls:
  add (call_QFU7V8TuWG02V5gfexXwc21g)
 Call ID: call_QFU7V8TuWG02V5gfexXwc21g
  Args:
    a: 40
    b: 12
  multiply (call_UaSA1yqt7eg2azBdzZPyXWx0)
 Call ID: call_UaSA1yqt7eg2azBdzZPyXWx0
  Args:
    a: 52
    b: 6
================================= Tool Message =================================
Name: multiply

312
================================== Ai Message ==================================

The result of adding 40 and 12 is 52. When you multiply that result by 6, you get 312.

And here's a joke for you:

Why don't scientists trust atoms?

Because they make up everything!
"""