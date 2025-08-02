from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from analyst import analyst_agent, AgentState 
import matplotlib

import chainlit as cl

from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage


def build_iframe(html: str, height: int = 600) -> str:
    """Return an iframe snippet with the folium HTML embedded via srcdoc."""
    import html as html_lib  # for escaping quotes
    quoted = html_lib.escape(html, quote=True)
    return (
        f'<iframe srcdoc="{quoted}" '
        f'style="width:100%; height:{height}px; border:none;"></iframe>'
    )

from chainlit import Text, Message

def send_map(html_page: str):
    iframe_snippet = build_iframe(html_page)
    return Message(
        content="🗺️ Interactive map:",
        elements=[ Text(name="map-iframe",
                        content=iframe_snippet,
                        display="inline") ]
    )

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


@cl.on_chat_start
async def on_chat_start():
    print("Session started")
    await cl.Message("Welcome to BoloChat! What would you like to do today?").send()


@cl.on_chat_end
async def on_chat_end():
    print("The user disconnected!")


@cl.on_message
async def on_message(msg: cl.Message):
    matplotlib.use("Agg") 

    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()

    await cl.Message(content="Thinking...").send()
    
    async for chunk, metadata in graph.astream({"messages": [HumanMessage(content=msg.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            chunk.content
            and isinstance(chunk, AIMessage)
            and metadata["langgraph_node"] == "supervisor"
            ):
            text = chunk.content[-1].get('text')
            await cl.Message(content=text, author="supervisor").send()

        elif isinstance(chunk, ToolMessage) and metadata["langgraph_node"] == "analyst_agent":
            if not chunk.artifact:
                tool_output = chunk.text()
                await cl.Message(
                    content=f"🔧 Tool output from analyst:\n```\n{tool_output}\n```",
                    author="tool",
                    type="tool"
                ).send()
            else:
                fig = chunk.artifact.get("image")
                html = chunk.artifact.get("html")

                # working
                '''if fig:
                    await cl.Message(
                        content="Here's the generated plot:",
                        elements=[cl.Pyplot(name="plot", figure=fig, display="inline", size="large")]
                    ).send()'''
                if html:    # not working
                    print("html detected!")
                    iframe = (
                        '<iframe srcdoc="' + html.escape(html, quote=True) +
                        '" style="width:100%;height:600px;border:none;"></iframe>'
                    )

                    await cl.Message(content=f"🗺️ Interactive map:\n\n{iframe}").send()


# also: if it detects fig AND html it shows a figure. Why?  

# https://www.datacamp.com/tutorial/chainlit