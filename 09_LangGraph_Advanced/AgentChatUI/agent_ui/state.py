import pandas as pd, geopandas as gpd
from typing import Union, Dict
from typing_extensions import Annotated
from langgraph.graph import MessagesState

from agent_ui.utils import merge_dictionary_entries

class DatasetState(MessagesState):
    loaded: Annotated[Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]], merge_dictionary_entries]
    descriptions: Annotated[Dict[str, str], merge_dictionary_entries]
    remaining_steps: int
