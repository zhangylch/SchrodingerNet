#! /usr/bin/env python3

import sys
import numpy as np
from src.params import *
import model.MPNN_det as MPNN
import model.MPNN_epsi as MPNN_epsi
import dataloader_pretrain.dataloader as dataloader
import src.print_info as print_info
import optax
from src.save_model import save_params, load_params
from src.DBOC import LocalEnergy
from src.print_params import print_params
from jax import vmap, jit
import folx



key = jax.random.split(key[-1], 2)

#==============================Equi MPNN==============================================================
model = MPNN.MPNN(emb_nl, MP_nl, det_nl, out_nl, numatom, nele, initene, spin_inp, neighlist, en_neighlist, nn_neighlist, index_ee, input_embed, cusp, index_l, index_i1, index_i2, ens_cg, index_add, index_cg, initbias_cg, init_contract, initbias_neigh, initbias_mp, initbias_det, initbias_embne, initbias_out, epson, sph_wf,  nn_table=nn_table, ndet=ndet, nwave=nwave, ncontract=ncontract, ngauss=ngauss, rmaxl=rmaxl, MP_loop=MP_loop)

model_epsi = MPNN_epsi.MPNN(emb_nl, MP_nl, det_nl, out_nl, numatom, nele, initene, spin_inp, neighlist, en_neighlist, nn_neighlist, index_ee, input_embed, cusp, index_l, index_i1, index_i2, ens_cg, index_add, index_cg, initbias_cg, init_contract, initbias_neigh, initbias_mp, initbias_det, initbias_embne, initbias_out, epson, sph_wf,  nn_table=nn_table, ndet=ndet, nwave=nwave, ncontract=ncontract, ngauss=ngauss, rmaxl=rmaxl, MP_loop=MP_loop)

params_rng = {"params": key[0]}
params = model.init(params_rng, coor[0], sqrt_mass)

vmap_model = jit(vmap(model.apply, in_axes=(None, 0, None)))

params_rng = {"params": key[0]}
params = model_epsi.init(params_rng, coor[0], sqrt_mass)

Hop = LocalEnergy(numatom, charge, sqrt_mass, model_epsi, sparsity=sparsity)

batch_le = jit(vmap(Hop, in_axes=(None, 0)))

def batch_map(batch_le):
    def _batch_map(params, coor):
        f_closure = lambda x: batch_le(params, x)
        local_values = jax.lax.map(f_closure, coor.reshape(-1, batchsize, numatom, 3))
        return local_values.reshape(-1)
    return jax.jit(_batch_map)
jit_batch_map = batch_map(batch_le)

dataloader = dataloader.Sampling(key[1], coor, sqrt_mass, ele_ion, params, model, vmap_model, nwalker=nwalker, MC_time=MC_time, step=step)

ferr=open("pre.err","w")
ferr.write("Equivariant MPNN package based on three-body descriptors \n")
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))

params = load_params(filename = "wf.pkl")
dataloader.params = params

def scan_fun(carry, _):
    params, coor, count = carry
    local_values = jit_batch_map(params, coor[count])
    clip_center = jnp.median(local_values)#params["params"]["elevel"]
    tv = jnp.mean(jnp.abs(local_values - clip_center))
    judge = jnp.logical_and(jnp.greater(local_values, clip_center - clip_scale * tv), jnp.less(local_values, clip_center+ clip_scale * tv))
    varene = jnp.dot(local_values, judge) / jnp.sum(judge)
    count +=1

    return (params, coor, count), varene

def get_values(params, coor):
    count = 0
    coor = coor.reshape(-1, nwalker, numatom, 3)
    (params, coor, count), store = jax.lax.scan(scan_fun, (params, coor, count), None, length=MC_time)
    return store

jit_scan = jax.jit(get_values)

varene = []
for iepoch in range(Epoch):
    coor = dataloader()
    tmp = jit_scan(params, coor)
    print(tmp)
    varene.append(tmp)
    sys.stdout.flush()
varene = jnp.concatenate(varene, axis=0)
np.savetxt("eval.txt", np.array(varene))
print(jnp.mean(varene[ninit:]))

ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
ferr.close()
print("Normal termination")
