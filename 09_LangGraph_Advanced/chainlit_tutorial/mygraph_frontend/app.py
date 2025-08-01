from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from analyst import analyst_agent, AgentState 

import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage

supervisor = create_supervisor(
    model=init_chat_model("openai:gpt-4.1-2025-04-14"),   
    agents=[analyst_agent],
    prompt=(
        "You are a supervisor managing a data analyst agent and a RAG agent spcialized in law retrieval. \n"
        "Assign data-analysis-related tasks to the data analyst\n"
        "Assign retrieval of laws or articles to the RAG agent.\n"
        "If the user asks to analize a law, you should first ask the RAG agent to retrieve it, then ask the data_analyst to analize it.\n\n"
        "Do not call agents in parallel, call one agent at a time."
    ),
    state_schema = AgentState,  
    add_handoff_back_messages=True,
    output_mode="full_history"
).with_config(tags=["main_model"])

graph = supervisor.compile()


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    for msg, metadata in graph.stream({"messages": [HumanMessage(content=msg.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        for msg, metadata in graph.stream({"messages": [HumanMessage(content=msg.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
            if msg.content:
                await final_answer.stream_token(msg.content)

        await final_answer.send()