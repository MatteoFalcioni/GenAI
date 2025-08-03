from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from analyst import analyst_agent, AgentState 
import matplotlib

import chainlit as cl

from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from pathlib import Path


# helper functions to get last html and last png from outputs folder

def get_html(folder = "./visualizer_outputs"):
    folder = Path(folder)          # or wherever you save them
    html_files = sorted(folder.glob("*.html"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True)
    if not html_files:
        return f"no html file found"
    
    html = html_files[0].read_text(encoding="utf-8")

    return html

def get_img(folder="./visualizer_outputs"):
    folder = Path(folder)
    png_files = sorted(folder.glob("*.png"),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True)
    if not png_files:
        return "No PNG file found"

    return str(png_files[0])


# build graph with create_supervisor()
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

# compile graph
graph = supervisor.compile(name="supervisor")


'''@cl.on_chat_start
async def on_chat_start():
    print("Session started")
    await cl.Message("Welcome to BoloChat! What would you like to do today?").send()'''


@cl.on_chat_end
async def on_chat_end():
    print("The user disconnected!")


@cl.on_message
async def on_message(msg: cl.Message):
    matplotlib.use("Agg")   # this should stop displaying with .show()

    config = {"configurable": {"thread_id": cl.context.session.id}, "recursion_limit" : 35}
    cb = cl.LangchainCallbackHandler()

    await cl.Message(content="Thinking...").send()
    
    async for chunk, metadata in graph.astream({"messages": [HumanMessage(content=msg.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):

        # problem: all images in a run are still in memory. If i ask for another plot, the previous one will be shown as well.   
        if (
            chunk.content
            and isinstance(chunk, AIMessage)
            and metadata["langgraph_node"] == "supervisor"
            ):
            text = chunk.content[-1].get('text')
            await cl.Message(content=text, author="supervisor").send()

        elif isinstance(chunk, ToolMessage) and metadata["langgraph_node"] == "analyst_agent":
            
            '''if not chunk.artifact:
                tool_output = chunk.content
                # tool_name = chunk.name
                await cl.Message(
                    content=f"🔧 Tool output from analyst:\n```\n{tool_output}\n```",
                    author="tool",
                    type="tool"
                ).send()
            else:'''

            # workarounds: streaming toolmessages as chunks conflicts with how i save images in toolmessages (i get same image / same html printed many times)
            # so i get the last png from folder instead. But if the model doesn't save it we're fucked, so fix this
            if chunk.artifact:
                
                fig = chunk.artifact.get("image")
                html = chunk.artifact.get("html")

                if fig:
                    print("image detected")

                    last_img_path = get_img()
                    image = cl.Image(path=last_img_path, name="Generated Plot", display="inline", size="large")

                    await cl.Message(
                        content="Generated plot:",
                        elements=[image]
                    ).send()

                if html:    
                    print("html detected!")

                    #retrieve last produced html 
                    last_html = get_html()  # limited: only gets the last html from folder

                    await cl.Message(
                        content="Interactive map:",
                        elements=[
                            cl.CustomElement(
                                name="HtmlElement",
                                props={"html": last_html},
                                display="inline",
                            )
                        ]
                    ).send()

                    



# also: if it detects fig AND html it shows a figure. Why?  

# https://www.datacamp.com/tutorial/chainlit