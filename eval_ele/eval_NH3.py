#! /usr/bin/env python3

# Grid spacing only; MC proposal widths are read from full_config.step.
rmin = 1.825             # bohr
rmax = 1.925
rstep = 0.1
xmin = 63.0            # degrees; N-H angle to +z, not the H-N-H angle
xmax = 119.0
xstep = 4.0
eq_points = [(1.91, 68.5), (1.91, 111.5)]
ts_points = [(1.88, 90.0)]
output_file = "NH3_PES.txt"
save_samples = True

import jax
import jax.numpy as jnp
import numpy as np
import model.MPNN_det as MPNN
from src.params import *
import dataloader.dataloader as dataloader
from src.data_config import ModelConfig
from src.hamiltonian_ele import LocalEnergy as Local_ele
from src.save_checkpoint import restore_checkpoint
from jax import vmap
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from src.create_sharding import create_sharding_pytree


def scan_points():
    r_values = np.round(np.arange(rmin, rmax + rstep/2, rstep), 10)
    x_values = np.round(np.arange(xmin, xmax + xstep/2, xstep), 10)
    points = [(r, x) for r in r_values for x in x_values]
    #for r, x in eq_points + ts_points:
    #    if not (rmin <= r <= rmax and xmin <= x <= xmax):
    #        raise ValueError("Extra geometry is outside the requested scan region.")
    #    if not any(abs(r-a) < 1e-10 and abs(x-b) < 1e-10 for a, b in points):
    #        points.append((r, x))
    return points


def nuclear_geometry(r, x):
    r = jnp.asarray(r, dtype=jnp_float)
    x = jnp.asarray(x, dtype=jnp_float)
    chi = jnp.deg2rad(x)
    phi = jnp.deg2rad(jnp.array([0.0, 120.0, 240.0], dtype=jnp_float))
    nuclei = jnp.zeros((4, 3), dtype=jnp_float)
    nuclei = nuclei.at[1:, 0].set(r * jnp.sin(chi) * jnp.cos(phi))
    nuclei = nuclei.at[1:, 1].set(r * jnp.sin(chi) * jnp.sin(phi))
    height = jnp.where(x == 90.0, jnp.zeros_like(r), r * jnp.cos(chi))
    nuclei = nuclei.at[1:, 2].set(height)
    return nuclei


if nele != 10 or npar != 14 or not np.allclose(np.asarray(charge[nele:]), [7, 1, 1, 1]):
    raise ValueError("Expected neutral NH3: 10 electrons followed by N, H, H, H.")
if full_config.step[1] != 0:
    raise ValueError("Fixed-geometry PES requires nuclear MC step=0 in full_config.step.")

ori_key = jax.random.split(key[-1], 4)

mesh = Mesh(jax.devices(), axis_names=('GPUs',))
scalar_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('GPUs'))
dtype_1 = jax.device_put(jnp.array(1.0, dtype=jnp_float), scalar_sharding)

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


Hop_ele = Local_ele(npar, charge, sqrt_mass, model,
                    dtype=full_config.jnp_dtype, sparsity=full_config.sparsity)
batch_ele = vmap(Hop_ele, in_axes=(None, 0))


def batch_map(batch_le):
    def _batch_map(params, coor):
        f_closure = lambda x: batch_le(params, x)
        local_values = jax.lax.map(f_closure, coor.reshape(-1, full_config.batchsize, npar, 3))
        return local_values.reshape(-1)
    return _batch_map


batch_ele = batch_map(batch_ele)
MC_sample = dataloader.Sampling(
    sqrt_mass, model_all.apply, MC_time=full_config.vmc_MC_time,
    step=full_config.step, jnp_dtype=full_config.jnp_dtype,
    nwalker=full_config.nwalker, prob=full_config.prob, nmove=full_config.nmove)


def init_sample(key, seedcoor, params, step, scale):
    def body(i, carry):
        key, seedcoor, init_logpsi, params, step = carry
        key, sample, init_logpsi, step = MC_sample(key, seedcoor, init_logpsi, params, step, scale)
        return key, sample, init_logpsi, params, step

    mwcoor = seedcoor * sqrt_mass[:, None]
    init_logpsi = vmap_model(params, mwcoor)
    key, seedcoor, init_logpsi, params, step = jax.lax.fori_loop(
        0, full_config.ninit_step, body, (key, seedcoor, init_logpsi, params, step))
    return key, seedcoor, step


def scan_fun(carry, _):
    key, params, seedcoor, init_logpsi, step = carry
    key, sample, init_logpsi, step = MC_sample(key, seedcoor, init_logpsi, params, step, dtype_1)
    ele_ene = batch_ele(params, sample)
    return (key, params, sample, init_logpsi, step), (ele_ene, init_logpsi)


def get_values(key, coor, params, step):
    mwcoor = coor * sqrt_mass[:, None]
    init_logpsi = vmap_model(params, mwcoor)
    (key, params, coor, init_logpsi, step), store = jax.lax.scan(
        scan_fun, (key, params, coor, init_logpsi, step), None, length=full_config.vmc_Epoch)
    ele_ene, init_logpsi = store
    ele_ene = jax.lax.all_gather(ele_ene, axis_name='GPUs')
    init_logpsi = jax.lax.all_gather(init_logpsi, axis_name='GPUs')
    varene = jnp.mean(ele_ene)
    return ele_ene, init_logpsi, varene, key, coor, step


init_sample = jax.jit(
              jax.shard_map(init_sample, mesh=mesh,
              in_specs=(P('GPUs'), P('GPUs'), P(), P(), P()),
              out_specs=(P('GPUs'), P('GPUs'), P()),
              check_vma=False
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

# Translate the initial electrons together with N, as in the density sampler.
seedcoor = seedcoor1 - seedcoor1[:, nele:nele+1, :]
seedcoor = jax.device_put(seedcoor, data_sharding)
step = jnp.array([full_config.step[0]]*nele + [full_config.step[1]]*(npar-nele), dtype=jnp_float)
step = jax.device_put(step/mass, scalar_sharding)

points = scan_points()
if jax.process_index() == 0:
    with open(output_file, "w") as f:
        f.write("r_bohr\tx_degree\tenergy_Ha")
    print("Number of geometries:", len(points), flush=True)

for i, (r, x) in enumerate(points):
    nuc_coor = jax.device_put(nuclear_geometry(r, x), scalar_sharding)
    seedcoor = seedcoor.at[:, nele:, :].set(nuc_coor)
    if jax.process_index() == 0:
        print(f"Point {i+1}/{len(points)}: r={r:.8f}, x={x:.8f}", flush=True)
    key, seedcoor, step = init_sample(key, seedcoor, params, step, dtype_1)
    ele_ene, init_logpsi, varene, key, seedcoor, step = get_values(key, seedcoor, params, step)
    if jax.process_index() == 0:
        with open(output_file, "a") as f:
            f.write(f"\n{r:.8f}\t{x:.8f}\t{float(varene):.12e}")
        if save_samples:
            fname = "eval_"+str(i)+".txt"
            ene_ens = jnp.stack([init_logpsi.reshape(-1), ele_ene.reshape(-1)], axis=1)
            np.savetxt(fname, np.array(ene_ens))
        print(varene, flush=True)

if jax.process_index() == 0:
    print("Normal termination", flush=True)
