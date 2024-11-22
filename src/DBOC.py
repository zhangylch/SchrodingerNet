"""Evaluating the Hamiltonian on a wavefunction."""


import folx
import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class LocalEnergy():
    
    def __init__(self, numatom, charge, sqrt_mass, MPNN, sparsity=6):
        self.allatomindex=jnp.arange(2 * int(numatom*(numatom-1)/2)).reshape(2, int(numatom*(numatom-1)/2))
        neigh=jnp.arange(numatom)
        num=0
        for i in range(numatom-1):
            self.allatomindex = self.allatomindex.at[0,num:num+numatom-1-i].set(i )           
            self.allatomindex = self.allatomindex.at[1,num:num+numatom-1-i].set(neigh[i+1:])
            num=num+numatom-1-i

        expand_charge = charge[self.allatomindex]
        self.mul_charge = expand_charge[0] * expand_charge[1]
        self.MPNN = MPNN.apply
        self.sparsity = sparsity
        self.sqrt_mass = sqrt_mass
        self.for_lap = self.batched_f()
        

    def __call__(self, params, coor):
        ke = self.LKE(params, coor)
        return ke


    def LKE(self, params, coor):
        mwcoor = coor * self.sqrt_mass[:, None]
        output = self.for_lap(params, mwcoor)
        result1 =  -jnp.sum(output[0][6:])/2.0 - jnp.sum(jnp.square(output[1][6:]))/2.0
    
        return result1

    def batched_f(self):
        
        def _batched_f(params, mwcoor):
            f_closure = lambda x: self.MPNN(params, x, self.sqrt_mass)
            for_lap = folx.LoopLaplacianOperator()(f_closure)
            return for_lap(mwcoor)
        return jax.jit(_batched_f)