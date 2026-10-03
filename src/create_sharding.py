from jax.sharding import NamedSharding
from jax.tree_util import tree_map
#  multiple-process
def create_sharding_pytree(pytree, mesh, spec):
    return tree_map(lambda x: NamedSharding(mesh, spec), pytree)

