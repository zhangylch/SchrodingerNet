from typing import Sequence, List, Union
import dataclasses
from jax import Array

#save the arguement for inference 
@dataclasses.dataclass
class PESConfig:
    npar: int
    nspec: int
    num_cg: int
    npaircode: int
    species: Array
    spec_indices: Array
    emb_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    radial_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    MP_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    out_nl: Sequence[Union[int, bool]]
    neighlist: Array
    count_l: Array
    index_l: Array
    index_i1: Array
    index_i2: Array
    ens_cg: Array
    index_add: Array
    index_den: Array
    index_squ: Array
    initbias_neigh: Array
    use_norm: bool
    use_bias: bool
    nwave: int
    nradial: int 
    rmaxl: int 
    prmaxl: int 
    MP_loop: int
    cst: float 
    cutoff: float

