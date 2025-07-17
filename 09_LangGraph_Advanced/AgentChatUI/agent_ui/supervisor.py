from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model

from agent_ui.state import DatasetState
from agent_ui.analyst_agent import analyst_agent
from agent_ui.load_env import set_if_undefined

set_if_undefined("OPENAI_API_KEY")
set_if_undefined("LANGSMITH_API_KEY")

supervisor_prompt = (
    "You are coordinating a data analyst \n\n"
    "When the data analyst finishes its work, you can end the workflow.\n"
    "Do not do any work yourself.\n"
)

supervisor = (
    create_supervisor(
      model=init_chat_model("openai:gpt-3.5-turbo"),
      agents=[analyst_agent],
      prompt=supervisor_prompt,
      state_schema=DatasetState,
      add_handoff_back_messages=True,
      output_mode="full_history",
    )
    .compile()
)
