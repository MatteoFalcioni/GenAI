from langchain_core.tools import tool
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage
import geopandas as gpd
import pandas as pd
import os
from pathlib import Path
from typing_extensions import Annotated
from typing import Union, Dict
from langchain_experimental.utilities import PythonREPL
from langgraph.types import Command
from langgraph.graph import MessagesState


# setup keys
import getpass
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")


DATASET_FOLDER = "./LLM_data"


def merge_dictionary_entries(existing_dict: Union[dict, None] = None, new_dict: Union[dict, None] = None) -> dict:
    """
    Custom reducer to merge dictionary updates:
    adds keys from new_dict only if they are not already in existing_dict.
    """
    if not existing_dict:
        existing_dict = {}
    if not new_dict:
        new_dict = {}

    for key, data in new_dict.items():
        if key not in existing_dict:
            existing_dict[key] = data
    
    return existing_dict

# custom state schema 
from langgraph.managed.is_last_step import RemainingSteps

class AgentState(MessagesState):
    loaded: Annotated[Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]], merge_dictionary_entries]
    remaining_steps: RemainingSteps     # key to let LangGraph automatically manage graph's supersteps


# ----------------------
# Tool: list datasets
# ----------------------
@tool
def list_loadable_datasets() -> str:
    """Lists all available parquet datasets in the dataset folder."""
    files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".parquet")]
    return "\n".join(files) if files else "No parquet datasets found."

@tool
def list_inmemory_datasets(
    state: Annotated[AgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Lists all loaded datasets and their type (DataFrame or GeoDataFrame)."""
    if not state["loaded"]:
        output = "No loaded datasets in memory. Use list_loadable_datasets() to see available files."
    
    else:
        lines = [
            f"- {name}: {'GeoDataFrame' if isinstance(df, gpd.GeoDataFrame) else 'DataFrame'} (shape={df.shape})"
            for name, df in state["loaded"].items()
        ]
        output = "\n".join(lines)

    return Command(update={
        "messages": [ToolMessage(content=output, tool_call_id=tool_call_id)],
    })


# ----------------------
# Tool: python repl
# ----------------------
repl = PythonREPL()
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute"], 
    state: Annotated[AgentState, InjectedState], 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Use this to execute python code. If you want to see the output of a value,
    print it out with `print(...)`. This is visible to the user. 
    """

    for name, df in state["loaded"].items():
        repl.globals[name] = df
    
    try:
        result = repl.run(code)
    except BaseException as e:
        tool_err_1 = f"Failed to execute. Error: {repr(e)}"
        return Command(update={"messages": [ToolMessage(content=tool_err_1, tool_call_id=tool_call_id)]})
    
    tool_output = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return Command(update={"messages": [ToolMessage(content=tool_output, tool_call_id=tool_call_id)]})


# ----------------------
# Tool: load datasets
# ----------------------
@tool
def load_dataset(file_name: str, 
                 tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Loads a Parquet dataset (optionally as GeoDataFrame) and updates state['loaded'][name].
    """
    update = {}

    file_stem = Path(file_name).stem
    file_name = f"{file_stem}.parquet"
    path = Path(DATASET_FOLDER) / file_name

    if not path.exists():
        available_files = os.listdir(DATASET_FOLDER)
        tool_err_result1 = f"File '{file_name}' not found. Available files: {available_files}"
        return Command(update={"messages": [ToolMessage(tool_err_result1, tool_call_id=tool_call_id)]})


    try:
        df = pd.read_parquet(path)
        if "geometry" in df.columns:
            try:
                df = gpd.read_parquet(path)
            except Exception as geo_err:
                tool_err_result2 = f"Geometry column found but failed to load as GeoDataFrame: {geo_err}"
                return Command(update={"messages": [ToolMessage(tool_err_result2, tool_call_id=tool_call_id)]})
            
        update[file_stem] = df

    except Exception as e:
        tool_err_result3 = f"Error loading dataset '{file_name}': {e}"
        return Command(update={"messages": [ToolMessage(tool_err_result3, tool_call_id=tool_call_id)]})


    return Command(update={
        "loaded": update,   
        "messages": [
            ToolMessage(f"Loaded dataset '{file_stem}' into memory.", tool_call_id=tool_call_id)
        ]
    })


# ----------------------
# Tool: describe_dataset
# ----------------------
@tool
def describe_dataset(name: str, 
                     state: Annotated[AgentState, InjectedState], 
                     tool_call_id: Annotated[str, InjectedToolCallId]
                     ) -> Command:
    """
    Generates a detailed description for a loaded dataset.

    This function returns a summary including:
    - the dataset type (DataFrame or GeoDataFrame),
    - a preview of the first few rows,
    - and the list of column names.
    """

    loaded = state.get('loaded')

    df = loaded.get(name)
    if df is None:
        loaded_keys = list(loaded.keys())
        available_data = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".parquet")]
        tool_err = f"Dataset '{name}' not found. \nLoaded datasets are: {loaded_keys} \nAvailable datasets to load are {available_data}"
        return Command(update={"messages": [ToolMessage(tool_err, tool_call_id=tool_call_id)]})

    dtype_str = type(df).__name__   # DataFrame or GeoDataFrame
    shape_info = f"{df.shape[0]} rows x {df.shape[1]} columns" # dataset shape
    # Show only the first N columns and limit long values
    MAX_COLS = 5
    MAX_ROWS = 5

    try:
        # Limit columns
        preview_df = df.iloc[:MAX_ROWS, :MAX_COLS].copy()
        # Truncate long values
        for col in preview_df.columns:
            preview_df[col] = preview_df[col].astype(str).str.slice(0, 40)

        head_str = preview_df.to_string(index=False)
    except Exception as e:
        tool_err = f"[Could not generate preview: {str(e)}]"
        return Command(update={"messages": [ToolMessage(content=tool_err, tool_call_id=tool_call_id)]})

    # show column types too
    cols_str = "\n".join([f"- {col} ({df[col].dtype})" for col in df.columns[:MAX_COLS]])
    if df.shape[1] > MAX_COLS:
        cols_str += f"\n...and {df.shape[1] - MAX_COLS} more columns"

    geometry_info = ""
    if dtype_str.lower() == "geodataframe":
        geometry_col = df.geometry.name if df.geometry.name in df.columns else None
        if geometry_col:
            geometry_info = f"\nActive geometry column: '{geometry_col}'"
        else:
            geometry_info = "\n No active geometry column set!"


    tool_output = (
        f"{dtype_str} | {shape_info}\n"
        f"{geometry_info}\n"
        f"---\n"
        f"Preview (first {MAX_ROWS} rows, {MAX_COLS} columns):\n{head_str}\n\n"
        f"Columns:\n{cols_str}"
    )

    return Command(
        update={
            "messages" : [ToolMessage(content=tool_output, tool_call_id=tool_call_id)],
        }
    )


# ----------------------
# Tool: fuzzy match name
# ----------------------
from rapidfuzz import process, fuzz

@tool
def fuzzy_match_name(dataset_name : str, 
                     dataset_column : str, 
                     input_str : str, 
                     state : Annotated[AgentState, InjectedState],
                     tool_call_id: Annotated[str, InjectedToolCallId],
                     threshold : int = 65,
) -> Command:
    """
    Performs fuzzy matching to find the best match for the input string
    within a specified column of a dataset.

    Returns the best matching string and score if above the threshold,
    otherwise a message indicating no match.

    The best match can then be used to extract data from that entry in the dataframe. 

    Example:
    - Input: ("points_of_interest", "name", "torre del orologio")
    - Output: "Torre dell'Orologio | score: 92"
    """
    loaded = state.get('loaded')

    try:
        known_names = (
            loaded[dataset_name][dataset_column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )
    except KeyError:
        tool_err = f"Dataset '{dataset_name}' or column '{dataset_column}' not found."
        return Command(update={"messages": [ToolMessage(tool_err, tool_call_id=tool_call_id)]})

    match_result = process.extractOne(
        input_str,
        known_names,
        scorer=fuzz.token_sort_ratio
    )

    if match_result is None:
        tool_err2 = f"No match candidates available in {dataset_name}.{dataset_column}."
        Command(update={"messages": [ToolMessage(tool_err2, tool_call_id=tool_call_id)]})

    match, score, _ = match_result
    if score >= threshold:
        tool_output = f"{match} | score: {score}"
    else:
        tool_output = f"No match found for '{input_str}' in '{dataset_name}.{dataset_column}' (best score: {score})"

    return Command(
        update={
            "messages": [ToolMessage(content=tool_output, tool_call_id=tool_call_id)]
        }
    )


analyst_suffix = (
    "You are a data analyst. Use your tools to explore and load datasets relevant to the task, analize them and then produce a visualization if requested.\n"
    "The files you need to load are in the subdirectory at ./LLM_data as .parquet files\n"
    "You can check which datasets are currently loaded with the `list_inmemory_datasets` tool, \
    and which datasets are available to load using the `list_loadable_datasets` tool.\n"
    "You can describe datasets with the `describe_dataset` tool.\n"
    "You can write custom python code with your `python_repl_tool`\n"
    "When asked to analize a law, use your `analize_law` tool. Laws are stored as graph state, so don't try to get them from datasets. Use your `analize_law` tool.\n\n"
    "**VERY IMPORTANT** : **When printing Python code, ALWAYS use `print(...)`**. Do NOT rely on implicit output like `quartieri.head()`. ALWAYS USE `print(...)`\n"
    "In your `python_repl_tool`, loaded datasets will appear as variables (e.g., if you load quartieri.parquet, the dataset will be accessible as `quartieri`)\n\n"
    "All spatial datasets use a geometry column (GeoDataFrame) containing shapely Point or Polygon objects.\n"
    "Always use the 'geometry' field when doing spatial operations, and avoid computing or reconstructing from latitude/longitude.\n"
    "When doing spatial queries (e.g., selecting features within 1 km), ensure you are working in a projected CRS (not WGS84). Use `.to_crs(epsg=32632)` to convert if needed.\n\n"
    "When matching column names in datasets, use the fuzzy_name_match() tool to first inspect what name the item is registered as in the dataframe.\n\n"
    "-------\n"
    "**Visualization**\n"
    "If visualization is requested, you must:\n"
    "   - Use the `python_repl_tool` to create **one clear, interpretable figure**, based on the request.\n"
    "   - display the visualization exactly ONCE using .show(), if possible.\n"
    "   - save the output figure to the `SAVING_DIRECTORY` folder, i.e. `./visualizer_outputs/`, after displaying it.\n"
    "Always aim to produce visually appealing plots. Your visualizations should be easy to interpret and presentation-ready."
    "\nDefault visualization preferences:\n"
    "- Use line plots, bar charts, or scatter plots for tabular data.\n"
    "- For geospatial data, use `.explore()` or overlay plots via geopandas or folium.\n\n"
)

from langgraph.prebuilt import create_react_agent

analyst_agent = create_react_agent(
    model="openai:gpt-4o",  
    tools=[list_loadable_datasets,
           list_inmemory_datasets, 
           load_dataset, 
           describe_dataset, 
           python_repl_tool,
           fuzzy_match_name,],
    prompt=analyst_suffix,
    name="analyst_agent",
    state_schema=AgentState
)