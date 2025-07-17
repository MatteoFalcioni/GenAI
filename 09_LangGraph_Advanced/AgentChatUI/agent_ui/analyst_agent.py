from langgraph.prebuilt import create_react_agent
from agent_ui.state import DatasetState
from agent_ui.tools import list_loadable_datasets, list_inmemory_datasets, load_dataset
import agent_ui.load_env 

prompt = (
    "You are a data analyst. Use your tools to explore and load datasets relevant to the task.\n"
    "The files you need to load are in the subdirectory at ../LLM_data\n"
    "Datasets are stored as `file_name.parquet`\n\n"
    "You can check which datasets are currently loaded with the `list_inmemory_datasets()` tool, \
    and which datasets are available to load using the `list_loadable_datasets()` tool.\n\n"
)

analyst_agent = create_react_agent(
    model="openai:gpt-4o-mini",
    tools=[list_loadable_datasets, list_inmemory_datasets, load_dataset],
    prompt=prompt,
    name="data_analyst",
    state_schema=DatasetState,
)
