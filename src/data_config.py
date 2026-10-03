from typing import Sequence, List, Union
import dataclasses
from jax import Array

#save the arguement for inference 
@dataclasses.dataclass
class ModelConfig:
    npar: int
    nele: int
    nspec: int
    ncom_spec: int
    num_cg: int
    num_cg_p: int
    nrealorb: int
    species: Array
    spec_indices: Array
    expand_indices: Array
    sqrt_mass: Array
    spin_inp: Array
    radial_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    den_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    MP_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    alpha_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    alphaee_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    orb_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    lorb_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    coeff_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    coeffee_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    emb_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    out_nl: Sequence[Union[int, bool]]
    nuc_nl: Sequence[Union[int, bool]]
    neighlist: Array
    en_neighlist: Array
    na_neighlist: Array
    ee_neighlist: Array
    nn_neighlist: Array
    index_l: Array
    index_i1: Array
    index_i2: Array
    ens_cg: Array
    index_add: Array
    index_den: Array
    index_squ: Array
    index_ee: Array
    cusp: Array
    initbias_neigh: Array
    initbias_out: Array
    S_index_l: Array
    S_index_add: Array
    S_index_add_p: Array
    S_index_den: Array
    S_index_den_p: Array
    use_norm: bool
    eee: bool
    neee:int
    nn_table: bool
    spin_table: int
    nwave: int
    nbasis: int
    ndet: int
    nradial: int 
    ngauss: int
    rorbl: int
    reorbl: int
    rmaxl: int 
    prmaxl: int 
    MP_loop: int
    en_eps: float
    ee_eps: float
    nn_eps_small: float
    nn_eps_large: float
    where_eps: float
    scale_nn: float
    scale_en: float
    scale_ee: float
    weight_hj: float
    cst: float 
    nup: int = 0

