#! /usr/bin/env python3

import sys
import numpy as np
from src.params import *
import model.MPNN_det as MPNN
import dataloader_pretrain.dataloader as dataloader
import src.print_info as print_info
import optax
from src.save_model import save_params, load_params
from src.hamiltonian_total import LocalEnergy
from src.print_params import print_params
from jax import vmap, jit
import folx



key = jax.random.split(key[-1], 2)

#==============================Equi MPNN==============================================================
model = MPNN.MPNN(emb_nl, MP_nl, det_nl, out_nl, numatom, nele, initene, spin_inp, neighlist, en_neighlist, nn_neighlist, index_ee, input_embed, cusp, index_l, index_i1, index_i2, ens_cg, index_add, index_cg, initbias_cg, init_contract, initbias_neigh, initbias_mp, initbias_det, initbias_embne, initbias_out, epson, sph_wf,  nn_table=nn_table, ndet=ndet, nwave=nwave, ncontract=ncontract, ngauss=ngauss, rmaxl=rmaxl, MP_loop=MP_loop)

params_rng = {"params": key[0]}
params = model.init(params_rng, coor[0], sqrt_mass)

vmap_model = jit(vmap(model.apply, in_axes=(None, 0, None)))

print("NN structure for wavefunction")
print_params(params)



Hop = LocalEnergy(numatom, charge, sqrt_mass, model, sparsity=sparsity)

batch_le = jit(vmap(Hop, in_axes=(None, 0)))

def batch_map(batch_le):
    def _batch_map(params, coor):
        f_closure = lambda x: batch_le(params, x)
        local_values = jax.lax.map(f_closure, coor.reshape(-1, batchsize, numatom, 3))
        return local_values.reshape(-1)
    return jax.jit(_batch_map)
jit_batch_map = batch_map(batch_le)



params = load_params(filename = "wf.pkl")
dataloader.params = params

coor = np.load("sample.npz")["coor"]
coor = jnp.array(coor)
print(coor)
nstart = jnp.mod(coor.shape[0], batchsize)
rcoor = coor[nstart:]
print(nstart, coor.shape[0])

local_values = jit_batch_map(params, rcoor)

np.savetxt("eval.txt", np.array(local_values))


mwcoor = coor * sqrt_mass[None, :, None]
logpsi = vmap_model(params, mwcoor, sqrt_mass)

np.savetxt("logpsi2.txt", np.array(2*logpsi))
