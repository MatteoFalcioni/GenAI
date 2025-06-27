"""
# Chatbot

This time we will:
    - use differnt message types (HumanMessage and AIMessage)
    - Mantain conversation history
    - Create a conversation loop
"""

from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv 

load_dotenv()


# AgentState
class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]   # the state is a list of human or AI messages


# initialize llm
llm = ChatOpenAI(model="gpt-4o")


# define llm node
def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""    
    
    # invoke llm
    response = llm.invoke(state['messages'])

    # append the answer to our messages to store it
    state['messages'].append(AIMessage(content=response.content))

    print(f"\nAI: {response.content}")

    print("Current State: ", state["messages"])
    
    return state


# construct graph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


# memory
conversation_history = []
memory_window = 10 # remember up to 10 messages ago (otherwise we may use too many tokens)


user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))   # append our message to the conversation (which is the state)
    result = agent.invoke({"messages": conversation_history})   # invoke the agent with the full conversation history
    conversation_history = result["messages"]   # update conversation history
    if len(conversation_history) > memory_window:
        conversation_history = conversation_history[-memory_window:]    # keep only last memory_window messages
    user_input = input("Enter: ")


# logging example (in order to keep convos) -> this would be better done in a vector database, but for now let's keep it simple
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")


"""
-------------------------------------------
OUTPUT:

matteo@pcmatteo:~/Learning_AI/GenAI/08_LangGraph$ python 0.8.9_Agent_2.py 
Enter: Hi my name is Matteo, how are you? 

AI: Hi Matteo! I'm an AI, so I don't have feelings in the same way humans do, but I'm here and ready to help you with whatever you need. How can I assist you today?
Current State:  [HumanMessage(content='Hi my name is Matteo, how are you? ', additional_kwargs={}, response_metadata={}), AIMessage(content="Hi Matteo! I'm an AI, so I don't have feelings in the same way humans do, but I'm here and ready to help you with whatever you need. How can I assist you today?", additional_kwargs={}, response_metadata={})]
Enter: What was my name? 

AI: Your name is Matteo.
Current State:  [HumanMessage(content='Hi my name is Matteo, how are you? ', additional_kwargs={}, response_metadata={}), AIMessage(content="Hi Matteo! I'm an AI, so I don't have feelings in the same way humans do, but I'm here and ready to help you with whatever you need. How can I assist you today?", additional_kwargs={}, response_metadata={}), HumanMessage(content='What was my name? ', additional_kwargs={}, response_metadata={}), AIMessage(content='Your name is Matteo.', additional_kwargs={}, response_metadata={})]
Enter: exit
Conversation saved to logging.txt

-------------------------------------------
"""