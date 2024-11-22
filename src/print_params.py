import jax
from flax.core import freeze, unfreeze

def print_params(params, parent_key=""):
    param_dict = unfreeze(params)
    for key, value in param_dict.items():
        full_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(value, dict):  # Check if it's a nested dictionary (FrozenDict)
            print_params(value, full_key)  # Recursively print nested keys
        else:
            print(f"Parameter Name: {full_key}, Shape: {value.shape}")

