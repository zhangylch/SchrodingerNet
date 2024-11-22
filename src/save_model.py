import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import pickle


# Saving the parameters to a file
def save_params(params, filename='model_params.pkl'):
    with open(filename, 'wb') as f:
        # Use flax.serialization to convert params to bytes and save with pickle
        bytes_data = flax.serialization.to_bytes(params)
        pickle.dump(bytes_data, f)

# Restoring the parameters from a file
def load_params(filename='model_params.pkl'):
    with open(filename, 'rb') as f:
        bytes_data = pickle.load(f)
        # Deserialize the parameters
        params = flax.serialization.from_bytes(None, bytes_data)
    return params

## Save the model parameters to a file
#save_params(params, 'model_params.pkl')
#
## Load the model parameters from the file
#restored_params = load_params('model_params.pkl')
#
## Verify that the restored params match the original params
#print(jax.tree_util.tree_all(jax.tree_map(lambda x, y: jnp.array_equal(x, y), params, restored_params)))  # Should print True

