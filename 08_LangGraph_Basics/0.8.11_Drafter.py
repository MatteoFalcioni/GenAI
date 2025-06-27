from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

load_dotenv()

# This is the global variable to store document content
document_content = ""

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"


    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    # it there have been no messages yet:
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    # invoke llm with all messages
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""

    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end" # goes to the end edge which leads to the endpoint
        
    return "continue"


def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# graph
graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", "agent")
graph.add_conditional_edges(source="tools",
                            path=should_continue,
                            path_map={
                                "continue" : "agent",
                                "end" : END
                            })
app = graph.compile()


def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()



""" 
OUTPUT: 

(openAI) (CNNLidar) matteo@pcmatteo:~/Learning_AI/GenAI/08_LangGraph$ python 0.8.11_Drafter.py 
 
 ===== DRAFTER =====

🤖 AI: I have tools available to update or save your document. Please let me know how you'd like to proceed or what content you'd like to update or add.

What would you like to do with the document? Write me a draft email to tell a collegue i can't make it into work today because im sick  

👤 USER: Write me a draft email to tell a collegue i can't make it into work today because im sick

🤖 AI: Here's a draft email for you:

---

Subject: Unable to Attend Work Today

Hi [Colleague's Name],

I hope this message finds you well. I wanted to let you know that I'm not feeling well today and, unfortunately, won't be able to make it into the office. 

I apologize for any inconvenience this may cause and will ensure to catch up on any missed work once I'm feeling better. Please let me know if there's anything urgent that needs my attention today.

Thank you for your understanding.

Best regards,

[Your Name]

--- 

Let me know if you need any changes or if you'd like to save this draft!

What would you like to do with the document? ok mi collegue name is Tom and my name is Matteo 

👤 USER: ok mi collegue name is Tom and my name is Matteo 

🤖 AI: 
🔧 USING TOOLS: ['update']

🛠️ TOOL RESULT: Document has been updated successfully! The current content is:
Subject: Unable to Attend Work Today

Hi Tom,

I hope this message finds you well. I wanted to let you know that I'm not feeling well today and, unfortunately, won't be able to make it into the office. 

I apologize for any inconvenience this may cause and will ensure to catch up on any missed work once I'm feeling better. Please let me know if there's anything urgent that needs my attention today.

Thank you for your understanding.

Best regards,

Matteo

What would you like to do with the document? I like it, save it

👤 USER: I like it, save it

🤖 AI: 
🔧 USING TOOLS: ['save']

🛠️ TOOL RESULT: Document has been updated successfully! The current content is:
Subject: Unable to Attend Work Today

Hi Tom,

I hope this message finds you well. I wanted to let you know that I'm not feeling well today and, unfortunately, won't be able to make it into the office. 

I apologize for any inconvenience this may cause and will ensure to catch up on any missed work once I'm feeling better. Please let me know if there's anything urgent that needs my attention today.

Thank you for your understanding.

Best regards,

Matteo

💾 Document has been saved to: sick_leave_email_to_Tom.txt

🛠️ TOOL RESULT: Document has been saved successfully to 'sick_leave_email_to_Tom.txt'.
"""