#! /usr/bin/env python3

import sys
import numpy as np
from src.params import *
import model.MPNN_det as MPNN
import dataloader.dataloader as dataloader
import src.print_info as print_info
import optax
from src.hamiltonian_ele import LocalEnergy
from src.print_params import print_params
from src.save_checkpoint import save_checkpoint, restore_checkpoint
from jax import vmap
import folx
from src.data_config import ModelConfig
from dataclasses import replace, asdict
import json
from typing import Optional, Any
from optax import tree_utils as otu
from jax.tree_util import tree_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental import multihost_utils
from src.create_sharding import create_sharding_pytree

dtype_10 = jnp.array(10.0, dtype=jnp_float)
dtype_0 = jnp.array(0.0, dtype=jnp_float)
dtype_100 = jnp.array(100.0, dtype=jnp_float)
dtype_1 = jnp.array(1, dtype=jnp_float)
dtype_099 = jnp.array(0.999, dtype=jnp_energy)
dtype_01 = jnp.array(0.1, dtype=jnp_float)
dtype_001 = jnp.array(0.001, dtype=jnp_energy)
dtype_clip = jnp.array(full_config.vmc_clip_norm, dtype=jnp_float)
# train function
def train(key, params, optim, opt_state, loss_grad_fn, schedule_fn, MC_sample, vmap_model, mesh, scalar_sharding, data_sharding, params_sharding, opt_state_sharding, full_config, mass, npar, nele, jnp_float, seedcoor1, start_step):

    def train_loop(ncyc):

        def optimize_epoch(i, carry):

            key, params, opt_state, seedcoor, init_logpsi, loss_fn, eigene, step = carry

            key, sample, init_logpsi, step = MC_sample(key, seedcoor, init_logpsi, params, step)
            loss, varene, grads = loss_grad_fn(params, sample)

            grads = tree_map(lambda g: jnp.where(jnp.isfinite(g), g, dtype_0), grads)
            grads = jax.lax.pmean(grads, axis_name="GPUs")
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
           
            judge = jnp.logical_and(jnp.abs(eigene - varene) < jnp.abs(eigene), jnp.isfinite(varene))
            varene = jnp.where(judge, varene, eigene)
            eigene = eigene * dtype_099 + varene * dtype_001
            loss_fn = loss_fn + loss
            return key, params, opt_state, sample, init_logpsi, loss_fn, eigene, step
       

        def optimize_cyc(key, params, opt_state, seedcoor, eigene, step):
            
            mwcoor = seedcoor * sqrt_mass[:, None]
            init_logpsi = vmap_model(params, mwcoor)
            loss = jnp.array(0.0, dtype=jnp_energy)
            key, params, opt_state, seedcoor, init_logpsi, loss, eigene, step = jax.lax.fori_loop(0, ncyc, optimize_epoch, (key, params, opt_state, seedcoor, init_logpsi, loss, eigene, step))
            loss = jnp.sqrt(loss / ncyc)
            return key, params, opt_state, seedcoor, loss, eigene, step

        return jax.shard_map(
               optimize_cyc, mesh=mesh,
               in_specs=(P('GPUs'), P(), P(), P('GPUs'), P(), P()), 
               out_specs=(P('GPUs'), P(), P(), P('GPUs'), P(), P(), P()),
               check_vma=False
               )

    train_epoch = train_loop(full_config.vmc_ncyc)

    train_ens = jax.jit(
    train_epoch,
    in_shardings=(
        data_sharding,
        params_sharding,
        opt_state_sharding,
        data_sharding,
        scalar_sharding,
        scalar_sharding
    ),
    out_shardings=(
        data_sharding,
        params_sharding,
        opt_state_sharding,
        data_sharding,
        scalar_sharding,
        scalar_sharding,
        scalar_sharding
    )
    )

    global_size = jax.device_count()
    key = jax.random.split(key, full_config.nwalker)
    key = jax.device_put(key, data_sharding)
    seedcoor = jax.device_put(jnp.asarray(seedcoor1, dtype=jnp_float), data_sharding)
    step = jnp.array([full_config.vmc_step[0]]*nele + [full_config.vmc_step[1]] * (npar - nele), dtype=jnp_float)
    step = jax.device_put(step / mass, scalar_sharding)
   
    if jax.process_index()==0: print_err = print_info.Print_Info(ferr)
    
    nepoch = 0
    istep = 0
    best_loss = jnp.array(1e20, dtype=jnp_energy)
    eigene = jax.device_put(jnp.array(full_config.ineig, dtype=jnp_energy), scalar_sharding)
    while nepoch < full_config.vmc_Epoch: 
 
        key, params, opt_state, seedcoor, loss, eigene, step = train_ens(key, params, opt_state, seedcoor, eigene, step)    
   
        out_eig = jax.device_get(eigene)
        out_loss = jax.device_get(loss)
        if jax.process_index()==0:
            lr = schedule_fn(nepoch)
            print_err(nepoch, lr, out_eig, out_loss)

        nepoch = nepoch + full_config.vmc_ncyc

 
        if loss < best_loss * 1.1:

            if loss < best_loss:
                best_loss = loss

            global_coor = multihost_utils.process_allgather(seedcoor, tiled=True)
            gathered_coor = jax.device_get(global_coor)
            start_step = start_step + 1
            save_checkpoint(
                step=start_step, 
                params=params, 
                opt_state=opt_state, 
                config=config, 
                ckpt_dir=full_config.ckpath,
                max_to_keep=5  # <--- 只保留最近 5 个，自动删除更早的
            )
            if jax.process_index() == 0:
                full_config = replace(full_config, ineig=float(eigene))
                full_config = replace(full_config, inloss=float(loss))
                with open("full_config.json", "w") as f:
                    json.dump(asdict(full_config), f, indent=4)

                np.savez("seedcoor1", coor=gathered_coor.reshape(-1, npar, 3))
                print(f"Step {start_step}: Saved checkpoint")
                print(step)

        if (not jnp.isfinite(loss)):
            restored = restore_checkpoint(
                full_config.ckpath, 
                params_sharding,
                opt_state_sharding,
                target_dtype=jnp_float
            )
    
            loss = jax.device_put(best_loss, scalar_sharding)
            _, params, _, _ = restored
            coor = np.load("seedcoor1.npz")["coor"]
            seedcoor = jax.device_put(coor, data_sharding)
 
        sys.stdout.flush()
    

key = jax.random.split(key[-1], 3)

#==============================Equi MPNN==============================================================
config = ModelConfig(npar=npar, nele=nele, nup=nup, nspec=nspec, ncom_spec=ncom_spec, num_cg=num_cg, num_cg_p=num_cg_p, nrealorb=nrealorb, species=species, spec_indices=spec_indices, expand_indices=expand_indices, sqrt_mass=sqrt_mass, spin_inp=spin_inp, MP_nl=full_config.MP_nl, radial_nl=full_config.radial_nl, den_nl=full_config.den_nl, alpha_nl=full_config.alpha_nl, orb_nl=full_config.orb_nl, lorb_nl=full_config.lorb_nl, coeff_nl=full_config.coeff_nl, coeffee_nl=full_config.coeffee_nl, alphaee_nl=full_config.alphaee_nl, emb_nl=full_config.emb_nl, out_nl=full_config.out_nl, nuc_nl=full_config.nuc_nl, neighlist=neighlist, en_neighlist=en_neighlist, na_neighlist=na_neighlist, ee_neighlist=ee_neighlist, nn_neighlist=nn_neighlist, index_l=index_l, index_i1=index_i1, index_i2=index_i2, ens_cg=ens_cg, index_add=index_add, index_den=index_den, index_squ=index_squ, index_ee=index_ee, cusp=cusp, initbias_neigh=initbias_neigh, initbias_out=initbias_out, S_index_l=S_index_l, S_index_add=S_index_add, S_index_add_p=S_index_add_p, S_index_den=S_index_den, S_index_den_p=S_index_den_p, eee=full_config.eee, neee=full_config.neee, use_norm=full_config.use_norm, nn_table=nn_table, spin_table=full_config.spin_table, nwave=full_config.nwave, nbasis=full_config.nbasis, ndet=full_config.ndet, nradial=full_config.nradial, ngauss=full_config.ngauss, rorbl=rorbl, reorbl=reorbl, rmaxl=rmaxl, prmaxl=prmaxl, MP_loop=full_config.MP_loop, en_eps=full_config.en_eps, ee_eps=full_config.ee_eps, nn_eps_small=full_config.nn_eps_small, nn_eps_large=full_config.nn_eps_large, weight_hj=full_config.weight_hj, scale_nn=full_config.scale_nn, scale_en=full_config.scale_en, scale_ee=full_config.scale_ee, where_eps=full_config.where_eps, cst=jnp.array(1.67462, dtype=jnp_float))

model_all = MPNN.MPNN(config)

params_rng = {"params": key[0]}
mwcoor = coor[0] * sqrt_mass[:, None]
params = model_all.init(params_rng, mwcoor)



if jax.process_index()==0:
    print("NN structure for wavefunction")
    print_params(params)

vmap_model = vmap(model_all.apply, in_axes=(None, 0))

def model(params, ele_coor, nuc_coor):
    all_coor = jnp.concatenate([ele_coor, nuc_coor+dtype_0*ele_coor[0,0]], axis=0)
    return model_all.apply(params, all_coor)

Hop = LocalEnergy(npar, charge, sqrt_mass, model, dtype=full_config.jnp_dtype, sparsity=full_config.sparsity)

batch_le = vmap(Hop, in_axes=(None, 0))

def batch_map(batch_le):
    def _batch_map(params, coor):
        f_closure = lambda x: batch_le(params, x)
        local_values = jax.lax.map(f_closure, coor.reshape(-1, full_config.batchsize, npar, 3))
        return local_values.reshape(-1)
    return _batch_map

batch_local = batch_map(batch_le)

def make_wf_gradient(batch_local, vmap_model):

    def get_loss(params, mwcoor, diff):
        logpsi = vmap_model(params, mwcoor)
        return jnp.mean(diff*logpsi)
 
    get_grad = jax.grad(get_loss)

    def _loss_and_grad(params, coor):

        local_values = batch_local(params, coor)
        gathered = jax.lax.all_gather(local_values, axis_name='GPUs')
        clip_center = jnp.nanmedian(gathered)
        variance = jnp.abs(gathered - clip_center)
        tv = jnp.mean(variance)
        clip_values = jnp.clip(local_values, clip_center - full_config.vmc_clip_scale * tv, clip_center + full_config.vmc_clip_scale * tv)
        varene = jax.lax.pmean(jnp.mean(local_values), axis_name="GPUs")
        clip_mean = jax.lax.pmean(jnp.mean(clip_values), axis_name="GPUs")
        loss = jax.lax.pmean(jnp.mean(jnp.square(clip_values - clip_mean)), axis_name="GPUs")
        mwcoor = coor * sqrt_mass[:, None]

        diff = clip_values - clip_mean
        diff = jax.lax.stop_gradient(diff)
        grads = get_grad(params, mwcoor, diff)
        return loss, varene, grads
        

    return _loss_and_grad

loss_grad_fn = make_wf_gradient(batch_local, vmap_model)

MC_sample = dataloader.Sampling(sqrt_mass, model_all.apply, MC_time=full_config.vmc_MC_time, nwalker=full_config.nwalker, jnp_dtype=full_config.jnp_dtype, step=full_config.vmc_step, prob=full_config.prob, nmove=full_config.nmove)

if jax.process_index()==0:
    ferr=open("VMC.err","w")
    ferr.write("Equivariant MPNN package based on three-body descriptors \n")
    ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))

schedule_fn=optax.schedules.warmup_cosine_decay_schedule(init_value=full_config.vmc_init_lr, peak_value=full_config.vmc_slr, warmup_steps=full_config.vmc_begin_step, decay_steps=full_config.vmc_patience_step, end_value=full_config.vmc_elr)

optim = optax.chain(optax.clip_by_global_norm(full_config.vmc_clip_norm), optax.adam(learning_rate=schedule_fn))

opt_state = optim.init(params)

mesh = Mesh(jax.devices(), axis_names=('GPUs',))
scalar_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('GPUs'))
params_sharding = create_sharding_pytree(params, mesh, P())
opt_state_sharding = create_sharding_pytree(opt_state, mesh, P())

params = jax.device_put(params, params_sharding)
opt_state = jax.device_put(opt_state, opt_state_sharding)

start_step = 0
save_checkpoint(
    step=start_step, 
    params=params, 
    opt_state=opt_state,
    config=config, 
    ckpt_dir=full_config.ckpath,
    max_to_keep=5  
)

if full_config.vmc_restart:
    restored = restore_checkpoint(
        full_config.ckpath, 
        params_sharding,
        opt_state_sharding,
        target_dtype=jnp_float
    )
    
    start_step, params, _, _ = restored
    

train(key[-1], params, optim, opt_state, loss_grad_fn, schedule_fn, MC_sample, vmap_model, mesh, scalar_sharding, data_sharding, params_sharding, opt_state_sharding, full_config, mass, npar, nele, jnp_float, seedcoor1, start_step)

if jax.process_index()==0:
    ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    ferr.close()
    print("Normal termination")
