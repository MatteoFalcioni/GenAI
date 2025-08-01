from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from analyst import analyst_agent, AgentState 

import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage

supervisor = create_supervisor(
    model=init_chat_model("anthropic:claude-sonnet-4-0"),   
    agents=[analyst_agent],
    prompt=(
        "You are coordinating a data analyst. He can analize data reguarding the city of Bologna\n\n"
        "Do not do any work yourself.\n"
        "You must only manage the workflow, greet the user and report what the workers do to the user."
    ),
    state_schema = AgentState,  
    add_handoff_back_messages=True,
    output_mode="full_history"
)

graph = supervisor.compile(name="supervisor")


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    for msg, metadata in graph.stream({"messages": [HumanMessage(content=msg.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and not isinstance(msg, ToolMessage)
            and not metadata["langgraph_node"] == "analyst_agent"
            ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()