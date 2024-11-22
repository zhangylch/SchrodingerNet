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
    det_nl: Sequence[Union[int, bool]]
    out_nl: Sequence[Union[int, bool]]
    numatom: int
    nele: int
    initene: jnp.ndarray 
    spin_inp: jnp.ndarray
    neighlist: jnp.ndarray
    en_neighlist: jnp.ndarray
    nn_neighlist: jnp.ndarray
    index_ee: jnp.ndarray
    input_embed: jnp.ndarray
    cusp: jnp.ndarray
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
    initbias_det: jnp.ndarray
    initbias_embne: jnp.ndarray
    initbias_out: jnp.ndarray
    epson: float
    sph_cal: Callable
    nn_table: bool = True
    ndet: int = 32
    nwave: int = 8
    ncontract: int = 128
    ngauss: int = 8
    rmaxl: int = 2
    MP_loop: int = 2

    def setup(self):

        self.elevel = self.param('elevel', lambda rng: self.initene)
        self.contracted_coeff = self.param('contracted_coeff', lambda rng: self.init_contract)

        self.emb_spin = MLP.MLP(num_output = self.ndet, num_blocks = self.emb_nl[0], features = self.emb_nl[1], layers_per_block = self.emb_nl[2], layer_norm = False, bias_init_value = jnp.ones(self.ndet))
        # define the embedded layer used to convert the atomin number of a coefficients
        self.emb_neighnn = MLP.MLP(num_output = self.initbias_neigh.shape[0], num_blocks = self.emb_nl[0], features = self.emb_nl[1], layers_per_block = self.emb_nl[2], layer_norm = False, bias_init_value = self.initbias_neigh)

        # Instantiate the NN class for the MPNN
        # create tge model for each iterations in MPNN
        self.emb_cg=MLP.MLP(num_output = self.initbias_cg.shape[0], num_blocks = self.emb_nl[0], features = self.emb_nl[1], layers_per_block = self.emb_nl[2], layer_norm = False, bias_init_value = self.initbias_cg)

        self.MPNN_list=[MLP.MLP(num_output = self.initbias_mp.shape[0], num_blocks = self.MP_nl[0], features = self.MP_nl[1], layers_per_block = self.MP_nl[2], layer_norm = self.MP_nl[3], bias_init_value = self.initbias_mp) for iMP_loop in range(self.MP_loop)]

        self.nn_det = MLP.MLP(num_output = self.initbias_det.shape[0], num_blocks = self.det_nl[0], features = self.det_nl[1], layers_per_block = self.det_nl[2], layer_norm = self.det_nl[3], bias_init_value = self.initbias_det)
        
        self.emb_ne=MLP.MLP(num_output = self.initbias_embne.shape[0], num_blocks = self.emb_nl[0], features = self.emb_nl[1], layers_per_block = self.emb_nl[2], layer_norm = False, bias_init_value = self.initbias_embne)
       
        self.outnn=MLP.MLP(num_output = self.initbias_out.shape[0], num_blocks = self.out_nl[0], features = self.out_nl[1], layers_per_block = self.out_nl[2], layer_norm = self.out_nl[3], bias_init_value = self.initbias_out)

    def __call__(self, cart, sqrt_mass):
        '''
        cart: jnp.float32/jnp.float64.
            represents the cartesian coordinates of systems with dimension 3*Natom. Natom is the number of atoms in the system.

        '''
        cart = cart / sqrt_mass[:, None]
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

        cg_coeff = self.emb_cg(self.input_embed).reshape(distances.shape[0], self.MP_loop+1, -1)[:, :, self.index_cg]
        for iter_loop, model in enumerate(self.MPNN_list):
            coeff = model(density.reshape(self.numatom, -1))
            center_orbital = self.interaction(radial_func[iter_loop+1], sph, center_orbital, coeff, cg_coeff[:,iter_loop])
            square_orbital = jnp.square(center_orbital)
            density = density.at[:, self.index_l].add(square_orbital)
            density = density / jnp.sqrt(2.0)
            
        # det
        input_en = self.input_embed[self.en_neighlist]
        dist_en = distances[self.en_neighlist]
        index_en = self.neighlist[0, self.en_neighlist]
        spin_func = self.emb_spin(self.spin_inp).reshape(self.nele, -1, self.ndet)
        ne_decay = self.emb_ne(input_en).reshape(-1, 2, self.nele, self.ndet)
        decay = ne_decay[:, 0] * jnp.exp(-jnp.abs(ne_decay[:, 1] * dist_en[:, None, None]))
        decay_en = jnp.zeros((self.nele, self.nele, self.ndet))
        decay_en = decay_en.at[index_en].add(decay)

        nnout = self.nn_det(density[:self.nele].reshape(self.nele, -1)).reshape(self.nele, self.nele, -1)

   
        det = jnp.einsum("ijk, ijk, ijk -> kij", nnout, decay_en, spin_func)
        # nuclear-nuclear orbital
        logdecay = jnp.sum(jnp.zeros(1))
        if self.nn_table:
            neighlist = self.neighlist[1][self.nn_neighlist]
            density = self.nn_interaction(radial_func[-1], sph, center_orbital, neighlist, cg_coeff[self.nn_neighlist, -1])
            nnout = self.outnn(density.reshape(neighlist.shape[0], -1))
            
            dist_nn = distances[self.nn_neighlist]
            logvalues = -jnp.sum(jnp.square(nnout[:, :self.ngauss] * dist_nn[:, None])  \
                      +jnp.square(jnp.divide(nnout[:, self.ngauss:2*self.ngauss], dist_nn[:, None])), axis=0)
            coeff = jnp.sum(nnout[:, self.ngauss*2:], axis=0)
            maxlog = jnp.max(logvalues)
            sublog = logvalues - maxlog
            values = jnp.dot(coeff, jnp.exp(sublog))
            abs_values = jnp.abs(values)
            logdecay = 2.0 * (maxlog + jnp.log(jnp.where(abs_values > self.epson, abs_values, self.epson)))

        cusp = jnp.sum(self.cusp * distances / (1.0 + distances))
            
        if self.nele < 1.5:
            psi = jnp.sum(det)
            max_det = 0.0
        else:
            sign, sdet_ele = jnp.linalg.slogdet(det)
            max_det = jnp.max(sdet_ele)
            sub_det = sdet_ele-max_det
            psi = jnp.einsum("i, i -> ", sign, jnp.exp(sub_det)) 

        abs_psi = jnp.abs(psi)
        logpsi = jnp.log(jnp.where(abs_psi > self.epson, abs_psi, self.epson)) + max_det + cusp
        return logpsi
 
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
        
    def nn_interaction(self, radial, sph, center_orbital, neighlist, cg_coeff):
        orbital = jnp.einsum("kij, ki -> kij", radial[self.index_l], sph)
        index_orbital1 = center_orbital[neighlist[:, None], self.index_i1]
        index_orbital2 = orbital[self.index_i2[:, None], self.nn_neighlist]
        inter_orbital = jnp.einsum("ikj, kij, ik, k -> ikj", index_orbital1, index_orbital2, cg_coeff, self.ens_cg)
        orbital = jnp.zeros((self.nn_neighlist.shape[0], orbital.shape[0], self.ncontract))
        orbital = orbital.at[:, self.index_add].add(inter_orbital)
        squ_orbital = jnp.square(orbital)
        density = jnp.zeros((neighlist.shape[0], self.rmaxl, self.ncontract))
        density = density.at[:,self.index_l].add(squ_orbital)
        return density
        
