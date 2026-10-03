#! /usr/bin/env python3

# All settings for the two nuclear coordinates are here.
rmin = 1.6             # bohr; histogram only, NOT a sampling boundary
rmax = 2.2
rstep = 0.01           # initial MC proposal width, not histogram bin size
xmin = 0.0            # degrees; histogram only, NOT a sampling boundary
xmax = 180.0
xstep = 1.0            # initial MC proposal width
rbin = 0.01           # histogram bin size, bohr
xbin = 1.0            # histogram bin size, degrees
r0 = None             # None: read initial r from seedcoor1[0]
x0 = 90.0             # all walkers start from the planar geometry
save_every = 200       # update density.txt every this many retained samples per walker
output = "nh3_density"

import os
import jax
import jax.numpy as jnp
import numpy as np
import model.MPNN_det as MPNN
from src.params import *
import dataloader_BO.dataloader as dataloader
from src.data_config import ModelConfig
from src.save_checkpoint import restore_checkpoint
from jax import vmap
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from src.create_sharding import create_sharding_pytree
from jax.experimental import multihost_utils

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

step_values = [full_config.step[0], rstep, xstep]
MC_sample = dataloader.Sampling(
    sqrt_mass, model_all.apply, MC_time=full_config.vmc_MC_time,
    step=step_values, jnp_dtype=full_config.jnp_dtype, nwalker=full_config.nwalker,
    prob=full_config.prob, nmove=full_config.nmove)


def init_sample(key, seedcoor, params, step, scale):
    def body(i, carry):
        key, seedcoor, init_logpsi, step = carry
        adapt_rate = (1.0 + i) ** -0.6
        key, sample, init_logpsi, step, x, r = MC_sample(
            key, seedcoor, init_logpsi, params, step, scale, adapt_rate)
        return key, sample, init_logpsi, step

    init_logpsi = vmap_model(params, seedcoor * sqrt_mass[:, None])
    key, seedcoor, init_logpsi, step = jax.lax.fori_loop(
        0, full_config.ninit_step, body, (key, seedcoor, init_logpsi, step))
    return key, seedcoor, step


def scan_fun(carry, i):
    key, params, seedcoor, init_logpsi, step = carry
    adapt_rate = (1.0 + i) ** -0.6
    key, sample, init_logpsi, step, x, r = MC_sample(
        key, seedcoor, init_logpsi, params, step, dtype_1, adapt_rate)
    # Rejected moves stay in the sample, with their repeated x and r.
    return (key, params, sample, init_logpsi, step), (x, r)


def get_values(key, coor, params, step, first):
    init_logpsi = vmap_model(params, coor * sqrt_mass[:, None])
    (key, params, coor, init_logpsi, step), (x, r) = jax.lax.scan(
        scan_fun, (key, params, coor, init_logpsi, step),
        first + jnp.arange(save_every))
    # Shard walkers, not time. Do not pmean the individual observations.
    return key, coor, step, x.T, r.T


init_sample = jax.jit(
    jax.shard_map(init_sample, mesh=mesh,
                  in_specs=(P('GPUs'), P('GPUs'), P(), P(), P()),
                  out_specs=(P('GPUs'), P('GPUs'), P()), check_vma=False),
    in_shardings=(data_sharding, data_sharding, params_sharding,
                  scalar_sharding, scalar_sharding),
    out_shardings=(data_sharding, data_sharding, scalar_sharding))

get_values = jax.jit(
    jax.shard_map(get_values, mesh=mesh,
                  in_specs=(P('GPUs'), P('GPUs'), P(), P(), P()),
                  out_specs=(P('GPUs'), P('GPUs'), P(), P('GPUs'), P('GPUs')),
                  check_vma=False),
    in_shardings=(data_sharding, data_sharding, params_sharding,
                  scalar_sharding, scalar_sharding),
    out_shardings=(data_sharding, data_sharding, scalar_sharding,
                   data_sharding, data_sharding))


def initial_coordinates(seedcoor):
    seedcoor = jnp.asarray(seedcoor, dtype=jnp_float)
    if seedcoor.shape != (full_config.nwalker, 14, 3):
        raise ValueError("Use seedcoor1.shape = (nwalker, 14, 3).")
    seedcoor = seedcoor - seedcoor[:, nele:nele+1, :]
    _, r_seed = MC_sample.get_xr(seedcoor)
    r_start = float(r_seed[0]) if r0 is None else r0
    x_start = x0
    if not (np.isfinite(r_start) and r_start > 0.0 and 0.0 < x_start < 180.0):
        raise ValueError("The initial geometry needs finite r0 > 0 and 0 < x0 < 180 degrees.")

    x = jnp.full((full_config.nwalker,), x_start, dtype=jnp_float)
    r = jnp.full((full_config.nwalker,), r_start, dtype=jnp_float)
    nuclei = vmap(MC_sample.nuclear_geometry)(r, x)
    return seedcoor.at[:, nele:, :].set(nuclei)


def bin_edges(low, high, width):
    if not (np.all(np.isfinite([low, high, width])) and low < high and width > 0):
        raise ValueError("Histogram edges need finite low < high and a positive bin width.")
    count = int(np.ceil((high-low) / width - 1e-10))
    edges = low + np.arange(count+1) * width
    edges[-1] = high
    return edges


if full_config.nwalker % jax.device_count():
    raise ValueError("nwalker must be divisible by the total number of devices.")
if full_config.vmc_Epoch < 1 or save_every < 1:
    raise ValueError("vmc_Epoch and save_every must be positive.")

key = jax.device_put(jax.random.split(ori_key[2], full_config.nwalker), data_sharding)
seedcoor = jax.device_put(initial_coordinates(seedcoor1), data_sharding)
step = jax.device_put(jnp.array(step_values, dtype=jnp_float), scalar_sharding)
key, seedcoor, step = init_sample(key, seedcoor, params, step, dtype_1)
print("test_1", flush=True)

r_edges = bin_edges(rmin, rmax, rbin)
x_edges = bin_edges(xmin, xmax, xbin)
r_values = (r_edges[:-1] + r_edges[1:]) / 2
x_values = (x_edges[:-1] + x_edges[1:]) / 2
rr, xx = np.meshgrid(r_values, x_values, indexing="ij")
hist = np.zeros((len(r_edges)-1, len(x_edges)-1), dtype=np.int64)
if jax.process_index() == 0:
    os.makedirs(output, exist_ok=True)
    previous_side = None
    observed_switches = np.zeros(full_config.nwalker, dtype=np.int64)

for first in range(0, full_config.vmc_Epoch, save_every):
    iteration = jax.device_put(jnp.array(full_config.ninit_step+first), scalar_sharding)
    key, seedcoor, step, x, r = get_values(key, seedcoor, params, step, iteration)
    print("test", flush=True)
    if jax.process_count() > 1:
        x, r = multihost_utils.process_allgather((x, r), tiled=True)
    if jax.process_index() == 0:
        count = min(save_every, full_config.vmc_Epoch-first)
        x = np.asarray(x).T[:count]
        r = np.asarray(r).T[:count]
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(r))):
            raise ValueError("Non-finite nuclear samples.")
        block_hist = np.histogram2d(r.ravel(), x.ravel(), bins=(r_edges, x_edges))[0]
        hist += block_hist.astype(np.int64)

        side = x > 90.0
        if previous_side is not None:
            observed_switches += side[0] != previous_side
        observed_switches += np.sum(side[1:] != side[:-1], axis=0)
        previous_side = side[-1]
        print("samples:", first+count, "step(e,r,x):", np.asarray(step),
              "fraction(x>90):", np.mean(side),
              "walkers observed on both sides:", np.count_nonzero(observed_switches), flush=True)

        total = (first+count) * full_config.nwalker
        outside_count = total - int(hist.sum())
        print("fraction inside histogram:", hist.sum() / total,
              "outside count:", outside_count, flush=True)
        area = np.diff(r_edges)[:, None] * np.diff(x_edges)[None, :]
        density = hist / (total * area)
        np.savetxt(os.path.join(output, "density.txt"),
                   np.column_stack([xx.ravel(), rr.ravel(), density.ravel(), hist.ravel()]),
                   fmt=["%.6f", "%.8f", "%.12e", "%d"],
                   header="x_degree r_bohr density_per_degree_per_bohr count")

if jax.process_index() == 0:
    if np.count_nonzero(observed_switches) < 0.1 * full_config.nwalker:
        print("WARNING: Few walkers crossed the plane; check mixing and convergence before interpreting the density.")
    print("Normal termination")
