import time
import numpy as np
import math
import os
import sys
import jax 
import jax.numpy as jnp
import src.read_data as read_data
from low_level import sph_cal, cg_cal
from src.gpu_sel import gpu_sel
from src.read_json import load_config
from dataclasses import replace, asdict
import json
from jax.experimental import sparse

full_config = load_config("full_config.json")

LOCAL_DEVICES=gpu_sel(full_config.local_size)

jax.distributed.initialize(local_device_ids=LOCAL_DEVICES)


jax.config.update("jax_default_matmul_precision", "highest")

if "32" in full_config.jnp_dtype:
    jax.config.update("jax_enable_x64", False)
    np_int = np.int32
    np_float = np.float32
    jnp_int = jnp.int32
    jnp_float = jnp.float32
elif "hy64" in full_config.jnp_dtype:
    np_int = np.int32
    np_float = np.float32
    jnp_int = jnp.int32
    jnp_float = jnp.float32
    jax.config.update("jax_enable_x64", True)
else:
    np_int = np.int64
    np_float = np.float64
    jnp_int = jnp.int64
    jnp_float = jnp.float64
    jax.config.update("jax_enable_x64", True)

# Keep energy arithmetic separate from parameter and coordinate precision.
jnp_energy = jnp.float32 if "32" in full_config.jnp_dtype else jnp.float64

def contract_sph(rmaxl, prmaxl, rorbl):

    index_i1 = []
    index_i2 = []
    cg_array = []
    index_den = []
    index_squ = []
    index_add = []
    count_l = jnp.zeros(prmaxl, dtype=jnp_float)
    num_coeff = 0
    num_coeff_p = 0
    num_den = 0
    num_den_p = 0
    nrealorb = 0
    for lf in range(prmaxl):
        for li1 in range(rmaxl):
            low = abs(li1 - lf)
            up = min(prmaxl, li1+lf+1)
            for li2 in range(low, up):
                cg = cg_cal.clebsch_gordan(li1, li2, lf)
                count_l = count_l.at[lf].add(1)        
                for mf in range(0, 2*lf+1):
                    dim3 = lf * lf + mf
                    index_add.append(dim3)
                    index_squ.append(num_coeff)

                    for mi1 in range(0, 2*li1+1):
                        for mi2 in range(0, 2*li2+1):
                            dim2 = li2 * li2 +  mi2 
                            dim1 = li1 * li1 +  mi1
                            if np.abs(cg[mi1, mi2, mf]) > 1e-3:
                                index_i1.append(dim1)
                                index_i2.append(dim2)
                                if lf < rorbl-0.5:
                                    nrealorb += 1

                                cg_array.append(cg[mi1, mi2, mf])
                                index_den.append(num_den)
                                if jax.process_index()==0: print(lf, mf, li1, mi1, li2, mi2, cg[mi1, mi2, mf])
                    num_den += 1
                    if lf < rorbl-0.5: num_den_p +=1
                num_coeff += 1
                if lf < rorbl-0.5: num_coeff_p +=1
         
    index_i1 = jnp.asarray(index_i1, dtype=jnp_int)
    index_i2 = jnp.asarray(index_i2, dtype=jnp_int)
    ens_cg = jnp.asarray(cg_array, dtype=jnp_float) #* expand_norm
    index_add = jnp.asarray(index_add, dtype=jnp_int)
    index_den = jnp.asarray(index_den, dtype=jnp_int)
    index_squ = jnp.asarray(index_squ, dtype=jnp_int)

    index_l = jnp.arange(rmaxl*rmaxl, dtype=jnp_int)
    for l in range(rmaxl):
        index_l = index_l.at[l*l:(l+1)*(l+1)].set(l)

    return index_i1, index_i2, ens_cg, index_add, index_den, index_squ, index_l, count_l, num_coeff, num_coeff_p, nrealorb, num_den_p



key = jax.random.PRNGKey(full_config.seed)

ele_ion = jnp.array(full_config.ele_ion)

initcoor, mass, charge, species = read_data.Read_data(datafloder=full_config.datapath, dtype=jnp_float)
sqrt_mass = jnp.sqrt(mass)
npar = species.shape[0]
nele = int(jnp.sum(jnp.less(species, 0.8)))
nup = int(jnp.sum(jnp.less(species[:nele], 0.0)))

neighlist = jnp.arange(2 * int(npar * (npar - 1)), dtype=jnp_int).reshape(2, int(npar * (npar - 1)))
num = 0
for i in range(npar):
    for j in range(npar):
        if abs(i - j) > 0.5:
             neighlist = neighlist.at[0, num].set(i)
             neighlist = neighlist.at[1, num].set(j)
             num += 1

judge = jnp.logical_and(jnp.less(neighlist[0], nele - 0.5), jnp.greater(neighlist[1], nele-0.5))
en_neighlist = jnp.nonzero(judge)[0].astype(jnp_int).reshape(-1)

judge = jnp.greater(neighlist[0], nele - 0.5)
na_neighlist = jnp.nonzero(judge)[0].astype(jnp_int).reshape(-1)

nn_neighlist = en_neighlist
if npar - nele > 1.5:
    judge = jnp.logical_and(jnp.greater(neighlist[0], nele - 0.5), \
                            jnp.greater(neighlist[1], nele - 0.5))
    nn_neighlist = jnp.nonzero(judge)[0].astype(jnp_int).reshape(-1)

ee_neighlist = en_neighlist
if nele > 1.5:
    judge = jnp.logical_and(jnp.less(neighlist[0], nele-0.5), \
                           jnp.less(neighlist[1], nele-0.5))
    ee_neighlist = jnp.nonzero(judge)[0].astype(jnp_int).reshape(-1)


rmaxl = full_config.maxl + 1
prmaxl = full_config.pmaxl + 1
rorbl = full_config.orbl + 1
reorbl = full_config.eorbl + 1

index_i1, index_i2, ens_cg, index_add, index_den, index_squ, index_l, count_l, num_cg, num_cg_p, nrealorb, num_den_p = \
    contract_sph(rmaxl, prmaxl, rorbl)

reduce_spec, spec_indices = jnp.unique(species, return_inverse=True)
spec_indices = spec_indices.astype(jnp_int).reshape(-1)
reduce_spec = reduce_spec.astype(jnp_float)

nspec = reduce_spec.shape[0]
expand_spec = jnp.transpose(species[neighlist], (1, 0, 2)).reshape(-1, 2)
com_spec, expand_indices = jnp.unique(expand_spec, axis=0, return_inverse=True)
expand_indices = expand_indices.astype(jnp_int)
ncom_spec = com_spec.shape[0]


spin_orb = jnp.array(full_config.spin_orb, dtype=jnp_float)
spin_inp = jnp.abs(species[:nele] + spin_orb).reshape(-1).astype(jnp_int)


nn_table = False
if npar - nele > 1.5:
    nn_table = True
nn_table = full_config.nn_table & nn_table

# initial cusp 
neigh_spec = species[neighlist[0]]
center_spec = species[neighlist[1]]
add_spec = neigh_spec + center_spec
mul_spec = neigh_spec * center_spec

cusp = -jnp.ones(neighlist.shape[1]) * jnp.abs(mul_spec).reshape(-1) * 2

if nele > 1.5:
    index_ee = jnp.abs(add_spec[ee_neighlist]).astype(jnp_int).reshape(-1)
    coeff = jnp.abs(add_spec[ee_neighlist]) + 1.0
    tmp = -cusp[ee_neighlist] / coeff.reshape(-1)
    cusp = cusp.at[ee_neighlist].set(tmp)

cusp = cusp / 2.0

# Keep the pair-type numbering and fill the diagonal with zero.
pair_index = np.full((npar, npar), npar*(npar-1), dtype=np_int)
pair_index[~np.eye(npar, dtype=bool)] = np.arange(npar*(npar-1), dtype=np_int)
expand_indices = jnp.concatenate([expand_indices.reshape(-1), jnp.zeros(1, dtype=expand_indices.dtype)])[pair_index]
cusp = jnp.concatenate([cusp.reshape(-1), jnp.zeros(1, dtype=cusp.dtype)])[pair_index]


key = jax.random.split(key, 7)

initbias_neigh = jax.random.normal(key[0], shape=(ncom_spec, full_config.nradial), dtype=jnp_float) * full_config.rden
initbias_neigh = 1.0 / (jnp.abs(initbias_neigh) + 0.02)
initbias_neigh = initbias_neigh.astype(dtype=jnp_float)
initbias_out = jax.random.normal(key[1], shape=(full_config.ngauss*2,), dtype=jnp_float) / ((npar - nele) * max(npar - nele - 1, 1)) 

sparsity = int(full_config.sparsity)
# get the sparse S
def build_S(segment_ids, num_data, num_segments):
    indices = jnp.stack([segment_ids, jnp.arange(num_data)], axis=1)
    values = jnp.ones(num_data, dtype=jnp_float)
    return sparse.BCOO((values, indices), shape=(num_segments, num_data))

S_index_l = build_S(index_l, rmaxl*rmaxl, rmaxl)
S_index_add = build_S(index_add, index_add.shape[0], prmaxl*prmaxl)
index_add_p = index_add[:num_den_p]
S_index_add_p = build_S(index_add_p, num_den_p, rorbl*rorbl)
#index_add_p = index_add[index_den[:nrealorb]]
#S_index_add_p = build_S(index_add_p, index_add_p.shape[0], rorbl*rorbl)

S_index_den = build_S(index_den, index_den.shape[0], index_add.shape[0])
S_index_den_p = build_S(index_den[:nrealorb], nrealorb, num_den_p)
S_index_l = S_index_l.todense()
S_index_add = S_index_add.todense()
S_index_add_p = S_index_add_p.todense()
S_index_den_p = S_index_den_p.todense()
S_index_den = S_index_den.todense()
norm = jnp.sqrt(count_l[index_l[:prmaxl*prmaxl]])
S_index_add_p = S_index_add_p / norm[:rorbl*rorbl, None]
S_index_add = S_index_add / norm[:, None]

eee = (nele > 1.5) & full_config.eee
full_config = replace(full_config, eee=eee)

numpoint = initcoor.shape[0]
with open("full_config.json", "w") as f:
    json.dump(asdict(full_config), f, indent=4)

try:
    coor = np.load("seedcoor1.npz")["coor"]
    seedcoor1 = jnp.array(coor.astype(np_float))
    coor = np.load("seedcoor2.npz")["coor"]
    seedcoor2 = jnp.array(coor.astype(np_float))
    if jax.process_index()==0: print("load data")

except FileNotFoundError:
    try:
        coor = np.load("seedcoor1.npz")["coor"]
        seedcoor1 = jnp.array(coor.astype(np_float))
        d_range = species[nele + ele_ion]
        nrepeat = math.ceil(full_config.nwalker/numpoint)
        coor = jnp.concatenate([initcoor] * nrepeat, axis=0)[:full_config.nwalker]
        offset = coor[:, ele_ion + nele]
        init_ele = jax.random.normal(key[4], (full_config.nwalker, nele, 3), dtype=jnp_float) * 2.0 / d_range + offset
        coor = coor.at[:,:nele].set(init_ele)
        seedcoor2 = coor

        if jax.process_index()==0: print("load seed data")
    except FileNotFoundError:

        d_range = species[nele + ele_ion]
        nrepeat = math.ceil(full_config.nwalker / full_config.vmc_numpoint)
        coor = jnp.concatenate([initcoor[:full_config.vmc_numpoint]] * nrepeat, axis=0)[:full_config.nwalker]
        offset = coor[:, ele_ion + nele]
        init_ele = jax.random.normal(key[4], (int(full_config.nwalker), nele, 3), dtype=jnp_float) * 2.0 / d_range + offset
        seedcoor1 = coor.at[:,:nele].set(init_ele)

        d_range = species[nele + ele_ion]
        nrepeat = math.ceil(full_config.nwalker / numpoint)
        coor = jnp.concatenate([initcoor] * nrepeat, axis=0)[:full_config.nwalker]
        offset = coor[:, ele_ion + nele]
        init_ele = jax.random.normal(key[5], (int(full_config.nwalker), nele, 3), dtype=jnp_float) * 2.0 / d_range + offset
        seedcoor2 = coor.at[:,:nele].set(init_ele)
        if jax.process_index()==0: print("random initilization")
if jax.process_index()==0: print("num_cg: {}, num_cg_p:{}".format(num_cg, num_cg_p))
