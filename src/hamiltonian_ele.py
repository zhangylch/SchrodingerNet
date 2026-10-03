"""Evaluating the Hamiltonian on a wavefunction."""


import folx
import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np


class LocalEnergy():
    
    def __init__(self, numatom, charge, sqrt_mass, MPNN, dtype="float32", sparsity=6):

        if "32" in dtype:
            self.jnp_float = jnp.float32
            self.jnp_int = jnp.int32
        else:
            self.jnp_float = jnp.float64
            self.jnp_int = jnp.int64

        self.allatomindex=jnp.arange(2 * int(numatom*(numatom-1)/2)).reshape(2, int(numatom*(numatom-1)/2))
        neigh=jnp.arange(numatom)
        num=0
        for i in range(numatom-1):
            self.allatomindex = self.allatomindex.at[0,num:num+numatom-1-i].set(i )           
            self.allatomindex = self.allatomindex.at[1,num:num+numatom-1-i].set(neigh[i+1:])
            num=num+numatom-1-i

        expand_charge = charge.astype(self.jnp_float)[self.allatomindex]
        self.mul_charge = expand_charge[0] * expand_charge[1]
        self.MPNN = MPNN
        self.sparsity = sparsity
        self.sqrt_mass = sqrt_mass
        self.lap_op = folx.ForwardLaplacianOperator(self.sparsity)
        self.for_lap = self.batched_f()
        self.nele = int(jnp.sum(sqrt_mass < 2.0))
        

    def __call__(self, params, coor):
        ke = self.LKE(params, coor)
        pot = self.get_pot(coor)
        return ke + pot


    def get_pot(self, coor):
        expand_coor = coor.astype(self.jnp_float)[self.allatomindex]
        distvec = expand_coor[1] - expand_coor[0]
        distances = jnp.linalg.norm(distvec, axis=1)
        pot = jnp.sum(jnp.true_divide(self.mul_charge, distances), dtype=self.jnp_float)
        return pot

    def LKE(self, params, coor):
        mwcoor = coor * self.sqrt_mass[:, None]
        output = self.for_lap(params, mwcoor)
        lap = output[0].astype(self.jnp_float)
        grad = output[1].astype(self.jnp_float)
        result1 = -(lap + jnp.sum(jnp.square(grad), dtype=self.jnp_float))/jnp.array(2.0, dtype=self.jnp_float)
    
        return result1

    def batched_f(self):
        
        def _batched_f(params, mwcoor):
            f_closure = lambda x: self.MPNN(params, x, mwcoor[self.nele:])
            for_lap = self.lap_op(f_closure)
            return for_lap(mwcoor[:self.nele])
        return _batched_f
