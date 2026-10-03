#! /usr/bin/env python3

import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

# Nuclear distances and grid spacing are in bohr.
r1min = 1.1
r1max = 3.5
r2min = 1.1
r2max = 3.5
rstep = 0.1
rsum_max = 6.0
ts_points = [(1.7488, 1.7488)]  # Table 3, DOI: 10.1021/acs.jpca.5c03032.
output_file = "H3_contour.txt"
save_samples = False

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

def scan_points():
    r1_values = np.round(np.arange(r1min, r1max + rstep/2, rstep), 10)
    r2_values = np.round(np.arange(r2min, r2max + rstep/2, rstep), 10)
    points = [(r1, r2) for r1 in r1_values for r2 in r2_values
              if r1 + r2 <= rsum_max + 1e-10]
    for r1, r2 in ts_points:
        if not (r1min <= r1 <= r1max and r2min <= r2 <= r2max
                and r1 + r2 <= rsum_max + 1e-10):
            raise ValueError("Transition-state point is outside the requested scan region.")
        if not any(abs(r1 - a) < 1e-10 and abs(r2 - b) < 1e-10 for a, b in points):
            points.append((r1, r2))
    return points

if nele != 3 or npar - nele != 3 or not np.allclose(np.asarray(charge[nele:]), 1.0):
    raise ValueError("Expected neutral H3: three electrons followed by three H nuclei.")

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

    return (key, params, sample, init_logpsi, step), (ele_ene, init_logpsi)

def get_values(key, coor, params, step):
    mwcoor = coor * sqrt_mass[:, None]
    init_logpsi = vmap_model(params, mwcoor)
    (key, params, coor, init_logpsi, step), store = jax.lax.scan(scan_fun, (key, params, coor, init_logpsi, step), None, length=full_config.vmc_Epoch)
    ele_ene, init_logpsi = store
    ele_ene = jax.lax.all_gather(ele_ene, axis_name='GPUs')
    init_logpsi = jax.lax.all_gather(init_logpsi, axis_name='GPUs')
    varene = jnp.mean(ele_ene)
    return ele_ene, init_logpsi, varene, key, coor, step

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
             out_specs=(P(), P(), P(), P('GPUs'), P('GPUs'), P()),
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
                 scalar_sharding,
                 scalar_sharding,
                 data_sharding,
                 data_sharding,
                 scalar_sharding
                 )
             )


key = jax.random.split(ori_key[2], full_config.nwalker)
key = jax.device_put(key, data_sharding)

seedcoor = jax.device_put(seedcoor1, data_sharding)
step = jnp.array([full_config.step[0]]*nele + [full_config.step[1]] * (npar - nele), dtype=jnp_float)
step = jax.device_put(step/mass, scalar_sharding)

points = scan_points()
if jax.process_index() == 0:
    with open(output_file, "w") as f:
        f.write("r1_bohr\tr2_bohr\tenergy_Ha")
    print("Number of geometries:", len(points), flush=True)
    if not ts_points:
        print("No extra literature transition-state point has been specified.", flush=True)

for i, (r1, r2) in enumerate(points):
    # H1 is the central nucleus; r1 = H1-H2, r2 = H1-H3.
    nuc_coor = jnp.array([[0.0, 0.0, 0.0], [-r1, 0.0, 0.0], [r2, 0.0, 0.0]], dtype=jnp_float)
    nuc_coor = jax.device_put(nuc_coor, scalar_sharding)
    seedcoor = seedcoor.at[:, nele:, :].set(nuc_coor)
    if jax.process_index() == 0:
        print(f"Point {i+1}/{len(points)}: r1={r1:.8f}, r2={r2:.8f}", flush=True)
    key, seedcoor, step = init_sample(key, seedcoor, params, step, dtype_1)
    ele_ene, init_logpsi, varene, key, seedcoor, step = get_values(key, seedcoor, params, step)
    if jax.process_index() == 0:
        with open(output_file, "a") as f:
            f.write(f"\n{r1:.8f}\t{r2:.8f}\t{float(varene):.12e}")
        if save_samples:
            fname = "eval_"+str(i)+".txt"
            ene_ens = jnp.stack([init_logpsi.reshape(-1), ele_ene.reshape(-1)], axis=1)
            np.savetxt(fname, np.array(ene_ens))
        print(varene, flush=True)
if jax.process_index() == 0:
    print("Normal termination", flush=True)
