# Standard library
import os
from pathlib import Path
from typing import Union
from typing_extensions import Annotated
# Third-party libraries
import pandas as pd
import geopandas as gpd
# LangChain and LangGraph
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
# dataset class
from state import DatasetState


DATASET_FOLDER = ".././LLM_data"

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


# ----------------------
# Tool: list datasets
# ----------------------
@tool
def list_loadable_datasets() -> str:
    """Lists all available parquet datasets in the dataset folder."""
    files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".parquet")]
    return "\n".join(files) if files else "No parquet datasets found."

@tool
def list_inmemory_datasets(state: Annotated[DatasetState, InjectedState]) -> str:
    """Lists all loaded datasets and their type (DataFrame or GeoDataFrame)."""
    if not state["loaded"]:
        return "No loaded datasets in memory. Use list_loadable_datasets() to see available files."
    
    lines = []
    for name, df in state["loaded"].items():
        dtype = "GeoDataFrame" if isinstance(df, gpd.GeoDataFrame) else "DataFrame"
        lines.append(f"- {name}: {dtype} (shape={df.shape})")

    return "\n".join(lines)

# ----------------------
# Tool: load datasets
# ----------------------
@tool
def load_dataset(file_name: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Loads a Parquet dataset (optionally as GeoDataFrame) and updates state['loaded'][name].
    """
    update = {}

    file_stem = Path(file_name).stem
    file_name = f"{file_stem}.parquet"
    path = Path(DATASET_FOLDER) / file_name

    if not path.exists():
        return f"File '{file_name}' not found."

    try:
        df = pd.read_parquet(path)
        if "geometry" in df.columns:
            try:
                df = gpd.read_parquet(path)
            except Exception as geo_err:
                return f"Failed to load as GeoDataFrame: {geo_err}"
        update[file_stem] = df
    except Exception as e:
        return f"Error loading dataset: {e}"

    return Command(update={
        "loaded": update,   
        "messages": [
            ToolMessage(f"Loaded dataset '{file_stem}' into memory.", tool_call_id=tool_call_id)
        ]
    })
