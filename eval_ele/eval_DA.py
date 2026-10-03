#! /usr/bin/env python3

import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import model.MPNN_det as MPNN
from src.params import *
import dataloader.dataloader as dataloader
from src.data_config import ModelConfig
from src.read_json import load_config
from src.hamiltonian_ele import LocalEnergy as Local_ele
from src.hamiltonian import LocalEnergy
from src.save_checkpoint import save_checkpoint, restore_checkpoint
import src.read_data as read_data
from jax import vmap, jit
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import folx
from src.create_sharding import create_sharding_pytree
from jax.experimental import multihost_utils

ori_key = jax.random.split(key[-1], 4)

mesh = Mesh(jax.devices(), axis_names=('GPUs',))
scalar_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('GPUs'))

dtype_2 = jax.device_put(jnp.array(2.0, dtype=jnp_float), scalar_sharding)
dtype_1 = jax.device_put(jnp.array(1.0, dtype=jnp_float), scalar_sharding)

#==============================Equi MPNN==============================================================
restored = restore_checkpoint(full_config.ckpath)

start_step, params, _, model_config = restored

params_sharding = create_sharding_pytree(params, mesh, P())
params = jax.device_put(params, params_sharding)
    

config = ModelConfig(**model_config)

model_all = MPNN.MPNN(config)

vmap_model = vmap(model_all.apply, in_axes=(None, 0))

def model(params, ele_coor, nuc_coor):
    all_coor = jnp.concatenate([ele_coor, nuc_coor], axis=0)
    return model_all.apply(params, all_coor)

Hop_ele = Local_ele(npar, charge, sqrt_mass, model, dtype=full_config.jnp_dtype, sparsity=full_config.sparsity)

batch_ele = vmap(Hop_ele, in_axes=(None, 0))

Hop = LocalEnergy(npar, charge, sqrt_mass, model_all.apply, dtype=full_config.jnp_dtype, sparsity=full_config.sparsity)

batch_le = vmap(Hop, in_axes=(None, 0))

def batch_map(batch_le):
    def _batch_map(params, coor):
        f_closure = lambda x: batch_le(params, x)
        local_values = jax.lax.map(f_closure, coor.reshape(-1, full_config.batchsize, npar, 3))
        return local_values.reshape(-1)
    return _batch_map

batch_ele = batch_map(batch_ele)
batch_local = batch_map(batch_le)

MC_sample = dataloader.Sampling(sqrt_mass, model_all.apply, MC_time=full_config.vmc_MC_time, step=full_config.step, jnp_dtype=full_config.jnp_dtype, nwalker=full_config.nwalker)

def init_sample(key, seedcoor, params, step, scale):
    def body(i, carry):
        key, seedcoor, init_logpsi, params, step = carry 
        key, sample, init_logpsi, step = MC_sample(key, seedcoor, init_logpsi, params, step, scale)
        return key, sample, init_logpsi, params, step

    mwcoor = seedcoor * sqrt_mass[:, None]
    init_logpsi = vmap_model(params, mwcoor)
    key, seedcoor, init_logpsi, params, step = jax.lax.fori_loop(0, full_config.ninit_step, body, (key, seedcoor, init_logpsi, params, step))
    return key, seedcoor, step

def scan_fun(carry, _):
    key, params, seedcoor, init_logpsi, step = carry
    key, sample, init_logpsi, step = MC_sample(key, seedcoor, init_logpsi, params, step, dtype_1)
    ele_ene = batch_ele(params, sample)

    return (key, params, sample, init_logpsi, step), ele_ene

def get_values(key, coor, params, step):
    mwcoor = coor * sqrt_mass[:, None]
    init_logpsi = vmap_model(params, mwcoor)
    (key, params, coor, init_logpsi, step), store = jax.lax.scan(scan_fun, (key, params, coor, init_logpsi, step), None, length=full_config.vmc_Epoch)
    ele_ene = store
    ele_ene = jax.lax.all_gather(ele_ene, axis_name='GPUs')
    varene = jnp.mean(ele_ene)
    return ele_ene, varene

init_sample = jax.jit(
              jax.shard_map(init_sample, mesh=mesh, 
              in_specs=(P('GPUs'), P('GPUs'), P(), P(), P()), 
              out_specs=(P('GPUs'), P('GPUs'), P())
              ),
              in_shardings=(
                  data_sharding,
                  data_sharding,
                  params_sharding,
                  scalar_sharding,
                  scalar_sharding
                  ),
              out_shardings=(
                  data_sharding,
                  data_sharding,
                  scalar_sharding
                  )
              )

get_values = jax.jit(
             jax.shard_map(get_values, mesh=mesh, 
             in_specs=(P('GPUs'), P('GPUs'), P(), P()), 
             out_specs=(P(), P()),
             check_vma=False
             ),
             in_shardings=(
                 data_sharding,
                 data_sharding,
                 params_sharding,
                 scalar_sharding
                 ),
             out_shardings=(
                 scalar_sharding,
                 scalar_sharding
                 )
             )


key = jax.random.split(ori_key[2], full_config.nwalker)
key = jax.device_put(key, data_sharding)

seedcoor = jax.device_put(seedcoor1, data_sharding)
step = jnp.array([full_config.step[0]]*nele + [full_config.step[1]] * (npar - nele), dtype=jnp_float)
step = jax.device_put(step/mass, scalar_sharding)

ens_plot = []
r = 3.629
for i in range(15):
    seedcoor = seedcoor.at[:, -1, 0].set(r)
    key, seedcoor, step = init_sample(key, seedcoor, params, step, dtype_1)
    ele_ene, varene = get_values(key, seedcoor, params, step)
    fname = "eval_"+str(i)+".txt"
    np.savetxt(fname, np.array(ele_ene.reshape(-1)))
    
    if jax.process_index()==0: print(r, varene, flush=True)
    r =r + 1.0
print("Normal termination")
