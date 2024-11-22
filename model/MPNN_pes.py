import sys
import numpy as np
import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
from jax.numpy import dtype,array
import flax.linen as nn
from typing import Sequence, List, Union, Callable
from low_level import MLP
from flax.core import freeze
from jax.lax import fori_loop




class MPNN(nn.Module):

    '''
    This MPNN module is designed for the calculation of equivariant message passing neural network for both periodic and molecular systems.
    Not only the invairant feature but also the eqivariant feature (sperical harmonic) are passed and progressively refined. 
    parameters:

    emb_nl: list
        defines the nn structure (only the hidden layers) of embedding neural network. Examples: [16,16]

    MP_nl: list
        defines the nn structure (only the hidden layers) used in each step except the last step of MPNN. Examples: [32,32]

    output_nl: list
        defines the nn structures of the last step of the MPNN (only the hidden layer), which will output the desired quantities, such as atomic energies or electron wave functions. Example: [64,64]

    nwave: int32/int64
         represents the number of gaussian radial functions. Example: 8

    max_l: int32/int64
         represents the maximal angular quantum numebr for the evaluation of spherical harmonic. Example: 2

    MP_loop: int32/int64
         represents the number of passing message in MPNN. Example: 3

    cutoff: float32/float64
         represents the cutoff radius for evaluating the local descriptors. Example: 4.0

    '''  

    emb_nl: Sequence[Union[int, bool]] # nblock, feature, nlayer
    MP_nl: Sequence[Union[int, bool]]
    out_nl: Sequence[Union[int, bool]]
    numatom: int
    neighlist: jnp.ndarray
    input_embed: jnp.ndarray
    index_l: jnp.ndarray
    index_i1: jnp.ndarray
    index_i2: jnp.ndarray
    ens_cg: jnp.ndarray
    index_add: jnp.ndarray
    index_cg: jnp.ndarray
    initbias_cg: jnp.ndarray
    init_contract: jnp.ndarray
    initbias_neigh: jnp.ndarray
    initbias_mp: jnp.ndarray
    initbias_out: jnp.ndarray
    sph_cal: Callable
    nwave: int = 8
    ncontract: int = 128
    rmaxl: int = 2
    MP_loop: int = 2

    def setup(self):

        self.contracted_coeff = self.param('contracted_coeff', lambda rng: self.init_contract)

        # define the embedded layer used to convert the atomin number of a coefficients
        self.emb_neighnn = MLP.MLP(num_output = self.initbias_neigh.shape[0], num_blocks = self.emb_nl[0], features = self.emb_nl[1], layers_per_block = self.emb_nl[2], layer_norm = False, bias_init_value = self.initbias_neigh)

        # Instantiate the NN class for the MPNN
        # create tge model for each iterations in MPNN
        self.emb_cg=MLP.MLP(num_output = self.initbias_cg.shape[0], num_blocks = self.emb_nl[0], features = self.emb_nl[1], layers_per_block = self.emb_nl[2], layer_norm = False, bias_init_value = self.initbias_cg)

        self.MPNN_list=[MLP.MLP(num_output = self.initbias_mp.shape[0], num_blocks = self.MP_nl[0], features = self.MP_nl[1], layers_per_block = self.MP_nl[2], layer_norm = self.MP_nl[3], bias_init_value = self.initbias_mp) for iMP_loop in range(self.MP_loop)]

        self.outnn=MLP.MLP(num_output = self.initbias_out.shape[0], num_blocks = self.out_nl[0], features = self.out_nl[1], layers_per_block = self.out_nl[2], layer_norm = self.out_nl[3], bias_init_value = self.initbias_out)

    def __call__(self, cart):
        '''
        cart: jnp.float32/jnp.float64.
            represents the cartesian coordinates of systems with dimension 3*Natom. Natom is the number of atoms in the system.

        '''
        expand_cart = cart[self.neighlist.reshape(-1)].reshape(2,-1,3)
        distvec = expand_cart[1] - expand_cart[0]
        distances = jnp.linalg.norm(distvec, axis=1)
        emb_coeff = self.emb_neighnn(self.input_embed)
        radial_func = jnp.exp(-jnp.square(emb_coeff[:, :self.nwave] * (distances[:, None] - emb_coeff[:, self.nwave:self.nwave*2])))
        radial_func = jnp.einsum("lkjm, im -> lkij", self.contracted_coeff, radial_func)
        sph = self.sph_cal(distvec.T/distances)

        iter_coeff = emb_coeff[:, self.nwave*2:].reshape(-1, self.rmaxl, self.ncontract)
        wradial = jnp.einsum("ikj, kij -> kij", iter_coeff, radial_func[0])
        worbital = jnp.einsum("kij, ki -> ikj", wradial[self.index_l], sph)
        center_orbital = jnp.mean(worbital.reshape(self.numatom, self.numatom-1, -1, self.ncontract), axis=1)
        square_orbital = jnp.square(center_orbital)
        density = jnp.zeros((self.numatom, self.rmaxl, self.ncontract))
        density = density.at[:, self.index_l].add(square_orbital)

        cg_coeff = self.emb_cg(self.input_embed).reshape(distances.shape[0], self.MP_loop, -1)[:, :, self.index_cg]
        for iter_loop, model in enumerate(self.MPNN_list):
            coeff = model(density.reshape(self.numatom, -1))
            center_orbital = self.interaction(radial_func[iter_loop+1], sph, center_orbital, coeff, cg_coeff[:,iter_loop])
            square_orbital = jnp.square(center_orbital)
            density = (density.at[:, self.index_l].add(square_orbital)) / jnp.sqrt(2.0)
            
        out = jnp.sum(self.outnn(density.reshape(self.numatom, -1)))
        return out
 
    def interaction(self, radial, sph, center_orbital, coeff, cg_coeff):
        coeff = coeff.reshape(self.numatom, self.rmaxl, self.ncontract)[self.neighlist[1]]
        wradial = jnp.einsum("kij, ikj ->kij", radial, coeff)
        orbital = jnp.einsum("kij, ki -> kij", wradial[self.index_l], sph)
        index_orbital1 = center_orbital[self.neighlist[1][:, None], self.index_i1]
        index_orbital2 = orbital[self.index_i2]
        inter_orbital = jnp.einsum("ikj, kij, ik, k -> kij", index_orbital1, index_orbital2, cg_coeff, self.ens_cg)
        worbital = jnp.zeros_like(orbital)
        worbital = worbital.at[self.index_add].add(inter_orbital)
        center_orbital = jnp.einsum("kimj -> ikj", worbital.reshape(-1, self.numatom, self.numatom-1, self.ncontract)) / (self.numatom-1)
        return center_orbital
        
