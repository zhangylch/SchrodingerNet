#! /usr/bin/env python3

import sys
import math
import numpy as np
from src.params import *
import model.MPNN_det as MPNN
import dataloader.dataloader as dataloader
import src.print_info as print_info
import optax
from src.hamiltonian import LocalEnergy
from src.print_params import print_params
from jax import vmap, jit
from src.save_checkpoint import save_checkpoint, restore_checkpoint
import folx
from jax.tree_util import tree_map
from flax.core import frozen_dict
from optax import tree_utils as otu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental import multihost_utils
from src.create_sharding import create_sharding_pytree
from src.data_config import ModelConfig

global_size = jax.device_count()
dtype_0 = jnp.array(0.0, dtype=jnp_float)
dtype_1 = jnp.array(1.0, dtype=jnp_float)
dtype_2 = jnp.array(2.0, dtype=jnp_float)
dtype_scale = jnp.array(full_config.clip_scale, dtype=jnp_energy)
dtype_10 = jnp.array(10.0, dtype=jnp_energy)
dtype_1e4 = jnp.array(1e4, dtype=jnp_energy)
dtype_099 = jnp.array(0.999, dtype=jnp_energy)
dtype_001 = jnp.array(0.001, dtype=jnp_energy)
dtype_1e_8 = jnp.array(1e-8, dtype=jnp_float)
nwalker_per_gpu = int(full_config.nwalker / global_size)


# train function
def train(key, params, optim, opt_state, grad_clip, clip_state, schedule_fn, grad_fn, mc_grad_fn, map_local, MC_sample, data_sample, vmap_model, mesh, scalar_sharding, data_sharding, params_sharding, opt_state_sharding, clip_state_sharding, full_config, mass, npar, nele, seedcoor1, seedcoor2, jnp_float, start_step):

    def train_loop(ncyc):

        def add_grad(tree1, tree2, weight):
            weight2 = weight / (dtype_1 + weight)
            weight1 = dtype_1 - weight2
            return tree_map(lambda g1, g2: weight1.astype(g1.dtype) * g1 + weight2.astype(g2.dtype) * g2, tree1, tree2)

        def body(i, carry):

            key, params, opt_state, clip_state, coor, seedcoor, init_logpsi, weight, ineig, inloss, step = carry
            key, sample, init_logpsi, step = MC_sample(key, seedcoor, init_logpsi, params, step)
            varene, mc_grads = mc_grad_fn(params, sample)
            varene = jnp.where(jnp.abs(varene - ineig) < jnp.abs(ineig), varene, ineig)
            ineig = ineig * dtype_099 + varene * dtype_001

            grads = grad_fn(params, coor[i], ineig, inloss)

            grads = tree_map(lambda g: jnp.where(jnp.isfinite(g), g, dtype_1e_8), grads)
            mc_grads = tree_map(lambda g: jnp.where(jnp.isfinite(g), g, dtype_1e_8), mc_grads)

            mc_grads, clip_state = grad_clip.update(jax.lax.pmean(mc_grads, axis_name="GPUs"), clip_state)
            grads, clip_state = grad_clip.update(jax.lax.pmean(grads, axis_name="GPUs"), clip_state)
            grads = add_grad(mc_grads, grads, weight)
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            return key, params, opt_state, clip_state, coor, sample, init_logpsi, weight, ineig, inloss, step
            
        def optimize_cyc(key, params, opt_state, clip_state, coor, mc_coor, weight, ineig, inloss, step):

            mwcoor = mc_coor * sqrt_mass[:, None]
            init_logpsi = vmap_model(params, mwcoor)
            key, params, opt_state, clip_state, coor, seedcoor, init_logpsi, weight, ineig, inloss, step = jax.lax.fori_loop(0, ncyc, body, (key, params, opt_state, clip_state, coor, mc_coor, init_logpsi, weight, ineig, inloss, step))

            return key, params, opt_state, seedcoor, ineig, step

        return optimize_cyc

    def sel_coor():
        def _sel(key, loss_out, div_val, coor):
            ulimit = jnp.minimum(loss_out * dtype_scale, dtype_10)
            judge = jnp.less(div_val, ulimit)
            div_val = jnp.where(judge, div_val, -jnp.inf)
            values, indices = jax.lax.top_k(div_val, full_config.nadd)
            return coor[indices]
        return _sel

    process_data = sel_coor() 
    train_epoch = train_loop(full_config.ncyc)

    def train_ens(key, seedcoor1, seedcoor2, pool_coor, nlocal, weight, ineig, inloss, params, opt_state, clip_state, step, vmc_step):


        shuffle_list = jax.random.randint(key[0], (full_config.ncyc, nwalker_per_gpu), 0, nlocal)
        train_coor = pool_coor[shuffle_list]
        key, params, opt_state, seedcoor1, ineig, vmc_step = train_epoch(key, params, opt_state, clip_state, train_coor, seedcoor1, weight, ineig, inloss, vmc_step)

        mwcoor = seedcoor2 * sqrt_mass[:, None]
        init_logpsi = vmap_model(params, mwcoor)
        key, sample, init_logpsi, step = data_sample(key, seedcoor2, init_logpsi, params, step)
        local_values = map_local(params, sample)
        tv = jax.lax.pmean(jnp.nanmean(jnp.abs(local_values - ineig)), axis_name="GPUs")
        clip_values = jnp.clip(local_values, ineig - full_config.clip_scale * tv, ineig + full_config.clip_scale * tv)

        NN_var = jnp.abs(clip_values - ineig)
        loss_out = jnp.mean(jnp.square(NN_var))
        loss_out = jnp.sqrt(jax.lax.pmean(loss_out, axis_name="GPUs"))
        loss_out = jnp.where(jnp.isfinite(loss_out), loss_out, inloss * dtype_1e4)

        NN_var = jnp.abs(local_values - ineig)
        sel_coor = process_data(key, loss_out, NN_var, sample)  
        pool_coor = jax.lax.dynamic_update_slice(pool_coor, sel_coor, (nlocal, 0, 0))

        nlocal = nlocal + full_config.nadd

        return key, seedcoor1, sample, pool_coor, nlocal, params, opt_state, loss_out, ineig, step, vmc_step

    def init_sample(key, seedcoor, params, step):
        def body(i, carry):
            key, seedcoor, init_logpsi, params, step = carry 
            key, sample, init_logpsi, step = data_sample(key, seedcoor, init_logpsi, params, step)
            return key, sample, init_logpsi, params, step

        mwcoor = seedcoor * sqrt_mass[:, None]
        init_logpsi = vmap_model(params, mwcoor)
        key, seedcoor, init_logpsi, params, step = jax.lax.fori_loop(0, full_config.ninit_step, body, (key, seedcoor, init_logpsi, params, step))
        return key, seedcoor, step


    train_ens = jax.jit(
                jax.shard_map(train_ens, mesh=mesh,
                in_specs=(P('GPUs'), P('GPUs'), P('GPUs'), P('GPUs'), P(), P(), P(), P(), P(), P(), P(), P(), P()),
                out_specs=(P('GPUs'), P('GPUs'), P('GPUs'), P('GPUs'), P(), P(), P(), P(), P(), P(), P()),
                check_vma=False
                ),
                in_shardings=(
                    data_sharding,
                    data_sharding,
                    data_sharding,
                    data_sharding,
                    scalar_sharding,
                    scalar_sharding,
                    scalar_sharding,
                    scalar_sharding,
                    params_sharding,
                    opt_state_sharding,
                    clip_state_sharding,
                    scalar_sharding,
                    scalar_sharding
                    ),
                out_shardings=(
                    data_sharding,
                    data_sharding,
                    data_sharding,
                    data_sharding,
                    scalar_sharding,
                    params_sharding,
                    opt_state_sharding,
                    scalar_sharding,
                    scalar_sharding,
                    scalar_sharding,
                    scalar_sharding
                    )
                )

    init_sample = jax.jit(
                  jax.shard_map(init_sample, mesh=mesh, 
                  in_specs=(P('GPUs'), P('GPUs'), P(), P()), 
                  out_specs=(P('GPUs'), P('GPUs'), P()),
                  check_vma=False
                  ),
                  in_shardings=(
                      data_sharding,
                      data_sharding,
                      params_sharding,
                      scalar_sharding
                      ),
                  out_shardings=(
                      data_sharding,
                      data_sharding,
                      scalar_sharding
                      )
                  )

    if jax.process_index()==0: print_err = print_info.Print_Info(ferr)

    best_loss = jnp.array(1e20, dtype=jnp_energy)
   
    key = jax.random.split(key, full_config.nwalker)
    key = jax.device_put(key, data_sharding)
    seedcoor1 = jax.device_put(jnp.asarray(seedcoor1, dtype=jnp_float), data_sharding)
    seedcoor2 = jax.device_put(jnp.asarray(seedcoor2, dtype=jnp_float), data_sharding)
    step = jnp.array([full_config.vmc_step[0]]*nele + [full_config.vmc_step[1]] * (npar - nele), dtype=jnp_float)
    vmc_step = jax.device_put(step / mass, scalar_sharding)
    step = jnp.array([full_config.step[0]]*nele + [full_config.step[1]] * (npar - nele), dtype=jnp_float)
    step = jax.device_put(step / mass, scalar_sharding)
    #  init sample to get ride of the unreasonable condiguration
    key, seedcoor2, step = init_sample(key, seedcoor2, params, step)

    nlocal = int(seedcoor1.shape[0] * 2 / global_size) 
    ntot = nlocal + int(math.ceil(full_config.Epoch / full_config.ncyc) * full_config.nadd)
    ntot = int(ntot / full_config.batchsize + 1) * full_config.batchsize

    def init_and_fill_fn(coor1, coor2):
        local_pool = jnp.zeros((ntot, npar, 3), dtype=jnp_float)
        
        current_n = coor1.shape[0]
        local_pool = local_pool.at[:current_n].set(coor1)
        local_pool = local_pool.at[current_n:2*current_n].set(coor2)
        
        return local_pool
    
    pool_coor = jax.shard_map(
        init_and_fill_fn,
        mesh=mesh,
        in_specs=(P('GPUs'), P('GPUs')), 
        out_specs=P('GPUs'),
        check_vma=False
    )(seedcoor1, seedcoor2)


    nlocal = jax.device_put(nlocal, scalar_sharding)
    varene = jax.device_put(jnp.array(full_config.ineig, dtype=jnp_energy), scalar_sharding)
    loss_out = jax.device_put(jnp.array(full_config.inloss, dtype=jnp_energy), scalar_sharding)
    weight = jax.device_put(jnp.array(full_config.init_weight, dtype=jnp_float), scalar_sharding)
    init_weight = jax.device_put(jnp.array(full_config.init_weight, dtype=jnp_float), scalar_sharding)
    final_weight = jax.device_put(jnp.array(full_config.final_weight, dtype=jnp_float), scalar_sharding)

    nepoch = 0
    while nepoch < full_config.Epoch: 

        key, seedcoor1, seedcoor2, pool_coor, nlocal, params, opt_state, loss_out, varene, step, vmc_step \
        = train_ens(key, seedcoor1, seedcoor2, pool_coor, nlocal, weight, varene, loss_out, params, opt_state, clip_state, step, vmc_step)

        lr = schedule_fn(nepoch)
        out_eig = jax.device_get(varene)
        out_loss = jax.device_get(loss_out)
        if jax.process_index()==0:
            print_err(nepoch, lr, out_eig, out_loss)

        nepoch = nepoch + full_config.ncyc

        if nepoch > full_config.begin_step:
            weight = (init_weight + (lr - full_config.slr) / (full_config.elr - full_config.slr) * (final_weight - init_weight)).astype(jnp_float)

        if loss_out < best_loss * 1.1:
            if loss_out < best_loss:
                best_loss = loss_out

            global_coor = multihost_utils.process_allgather(seedcoor1, tiled=True)
            gathered_coor1 = jax.device_get(global_coor)
            global_coor = multihost_utils.process_allgather(seedcoor2, tiled=True)
            gathered_coor2 = jax.device_get(global_coor)
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
                full_config = replace(full_config, ineig=float(varene))
                full_config = replace(full_config, inloss=float(loss_out))
                with open("full_config.json", "w") as f:
                    json.dump(asdict(full_config), f, indent=4)

                np.savez("seedcoor1", coor=gathered_coor1.reshape(-1, npar, 3))
                np.savez("seedcoor2", coor=gathered_coor2.reshape(-1, npar, 3))
                print(f"Step {start_step}: Saved checkpoint")
                print("step", step)
                print("vmc step", vmc_step)

        if (not jnp.isfinite(loss_out)) or loss_out > best_loss * 5e3:
            nlocal = nlocal - full_config.ndel
            restored = restore_checkpoint(
                full_config.ckpath, 
                params_sharding,
                opt_state_sharding,
                target_dtype=jnp_float
            )
    
            loss_out = jax.device_put(best_loss, scalar_sharding)
            _, params, _, _ = restored
            coor = np.load("seedcoor1.npz")["coor"]
            seedcoor1 = jax.device_put(jnp.asarray(coor, dtype=jnp_float), data_sharding)
 
 
        sys.stdout.flush()

    
key = jax.random.split(key[-1], 2)
full_config = load_config("full_config.json")

#==============================Equi MPNN==============================================================
config = ModelConfig(npar=npar, nele=nele, nup=nup, nspec=nspec, ncom_spec=ncom_spec, num_cg=num_cg, num_cg_p=num_cg_p, nrealorb=nrealorb, species=species, spec_indices=spec_indices, expand_indices=expand_indices, sqrt_mass=sqrt_mass, spin_inp=spin_inp, MP_nl=full_config.MP_nl, radial_nl=full_config.radial_nl, den_nl=full_config.den_nl, alpha_nl=full_config.alpha_nl, orb_nl=full_config.orb_nl, lorb_nl=full_config.lorb_nl, coeff_nl=full_config.coeff_nl, coeffee_nl=full_config.coeffee_nl, alphaee_nl=full_config.alphaee_nl, emb_nl=full_config.emb_nl, out_nl=full_config.out_nl, nuc_nl=full_config.nuc_nl, neighlist=neighlist, en_neighlist=en_neighlist, na_neighlist=na_neighlist, ee_neighlist=ee_neighlist, nn_neighlist=nn_neighlist, index_l=index_l, index_i1=index_i1, index_i2=index_i2, ens_cg=ens_cg, index_add=index_add, index_den=index_den, index_squ=index_squ, index_ee=index_ee, cusp=cusp, initbias_neigh=initbias_neigh, initbias_out=initbias_out, S_index_l=S_index_l, S_index_add=S_index_add, S_index_add_p=S_index_add_p, S_index_den=S_index_den, S_index_den_p=S_index_den_p, eee=full_config.eee, neee=full_config.neee, use_norm=full_config.use_norm, nn_table=nn_table, spin_table=full_config.spin_table, nwave=full_config.nwave, nbasis=full_config.nbasis, ndet=full_config.ndet, nradial=full_config.nradial, ngauss=full_config.ngauss, rorbl=rorbl, reorbl=reorbl, rmaxl=rmaxl, prmaxl=prmaxl, MP_loop=full_config.MP_loop, en_eps=full_config.en_eps, ee_eps=full_config.ee_eps, nn_eps_small=full_config.nn_eps_small, nn_eps_large=full_config.nn_eps_large, weight_hj=full_config.weight_hj, scale_nn=full_config.scale_nn, scale_en=full_config.scale_en, scale_ee=full_config.scale_ee, where_eps=full_config.where_eps, cst=jnp.array(1.67462, dtype=jnp_float))

model = MPNN.MPNN(config)

params_rng = {"params": key[0]}
mwcoor = coor[0] * sqrt_mass[:, None]
params = model.init(params_rng, mwcoor)

vmap_model = vmap(model.apply, in_axes=(None, 0))


if jax.process_index()==0:
    print("NN structure for wavefunction")
    print_params(params)

# get the totat energy
Hop = LocalEnergy(npar, charge, sqrt_mass, model.apply, dtype=full_config.jnp_dtype, sparsity=full_config.sparsity)

batch_le = vmap(Hop, in_axes=(None, 0))

def batch_map(batch_le):
    def _batch_map(params, coor):
        f_closure = lambda x: batch_le(params, x)
        local_values = jax.lax.map(f_closure, coor.reshape(-1, full_config.batchsize, npar, 3))
        return local_values.reshape(-1)
    return _batch_map
map_local = batch_map(batch_le)

def make_gradient(batch_local):

    def get_loss(params, coor, ineig, inloss):
        local_values = batch_local(params, coor)
        clip_values = jnp.clip(local_values, ineig - full_config.clip_scale * inloss, ineig + full_config.clip_scale * inloss)
        loss = jnp.mean(jnp.square(clip_values - ineig))
        return loss

    grad_loss = jax.grad(get_loss)

    def _map_grad(params, coor, ineig, inloss):
        f_closure = lambda x: grad_loss(params, x, ineig, inloss)
        grads = jax.lax.map(f_closure, coor.reshape(-1, full_config.sbatchsize, npar, 3))
        return tree_map(lambda x: jnp.sum(x, axis=0), grads)
    return _map_grad
        
grad_fn = make_gradient(batch_le)

def make_mc_gradient(batch_local, vmap_model):

    def get_loss(params, mwcoor, diff):
        logpsi = vmap_model(params, mwcoor)
        return jnp.mean(diff*logpsi)
 
    get_grad = jax.grad(get_loss)

    def _grad(params, coor):

        local_values = batch_local(params, coor)
        gathered = jax.lax.all_gather(local_values, axis_name='GPUs')
        clip_center = jnp.nanmedian(gathered)
        tv = jnp.nanmean(jnp.abs(gathered - clip_center))
        clip_values = jnp.clip(local_values, clip_center - full_config.vmc_clip_scale * tv, clip_center + full_config.vmc_clip_scale * tv)
        varene = jax.lax.pmean(jnp.mean(local_values), axis_name="GPUs")
        clip_mean = jax.lax.pmean(jnp.mean(clip_values), axis_name="GPUs")

        mwcoor = coor * sqrt_mass[:, None]
        diff = clip_values - clip_mean
        diff = jax.lax.stop_gradient(diff)
        grads = get_grad(params, mwcoor, diff)
        return clip_mean, grads

    return _grad

mc_grad_fn = make_mc_gradient(map_local, vmap_model)


data_sample = dataloader.Sampling(sqrt_mass, model.apply, MC_time=full_config.MC_time, nwalker=full_config.nwalker, jnp_dtype = full_config.jnp_dtype, step=full_config.step, prob=full_config.prob, nmove=full_config.nmove)

MC_sample = dataloader.Sampling(sqrt_mass, model.apply, MC_time=full_config.vmc_MC_time, nwalker=full_config.nwalker, jnp_dtype=full_config.jnp_dtype, step=full_config.vmc_step, prob=full_config.prob, nmove=full_config.nmove)

if jax.process_index()==0:
    ferr=open("wf.err","w")
    ferr.write("Wavefunction based on NEMP \n")
    ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))


schedule_fn=optax.schedules.warmup_cosine_decay_schedule(init_value=full_config.init_lr, peak_value=full_config.slr, warmup_steps=full_config.begin_step, decay_steps=full_config.patience_step, end_value=full_config.elr)

grad_clip = optax.clip_by_global_norm(full_config.clip_norm)
optim = optax.adam(learning_rate = schedule_fn)

opt_state = optim.init(params)
clip_state = grad_clip.init(params)

mesh = Mesh(jax.devices(), axis_names=('GPUs',))
scalar_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('GPUs'))
params_sharding = create_sharding_pytree(params, mesh, P())
opt_state_sharding = create_sharding_pytree(opt_state, mesh, P())
clip_state_sharding = create_sharding_pytree(clip_state, mesh, P())

params = jax.device_put(params, params_sharding)
opt_state = jax.device_put(opt_state, opt_state_sharding)
clip_state = jax.device_put(clip_state, clip_state_sharding)

start_step = 0
save_checkpoint(
    step=start_step, 
    params=params, 
    opt_state=opt_state,
    config=config, 
    ckpt_dir=full_config.ckpath,
    max_to_keep=5  
)

if full_config.restart:
    restored = restore_checkpoint(
        full_config.ckpath, 
        params_sharding,
        opt_state_sharding,
        target_dtype=jnp_float
    )
    
    start_step, params, _, _ = restored
    

train(key[-1], params, optim, opt_state, grad_clip, clip_state, schedule_fn, grad_fn, mc_grad_fn, map_local, MC_sample, data_sample, vmap_model, mesh, scalar_sharding, data_sharding, params_sharding, opt_state_sharding, clip_state_sharding, full_config, mass, npar, nele, seedcoor1, seedcoor2, jnp_float, start_step)

if jax.process_index()==0:
    ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
    ferr.close()
    print("Normal termination")
