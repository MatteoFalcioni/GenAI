from typing import Union

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