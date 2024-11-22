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


# train function
def train(params, optim, loss_grad_fn, schedule_fn, dataloader, Epoch):

    def outer_loop(nstep):

        def optimize_epoch(params, opt_state, sample, loss_fn, eigene):
            def body(i, carry):
                params, opt_state, coor, loss_fn, eigene = carry
                coor = coor.reshape(-1, nwalker, numatom, 3)
                loss, eig, grads = loss_grad_fn(params, coor[i])
                updates, opt_state = optim.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                loss_fn += loss
                eigene += eig
                return params, opt_state, coor.reshape(-1, numatom, 3), loss_fn, eigene
            return jax.lax.fori_loop(0, nstep, body, (params, opt_state, sample, loss_fn, eigene))

        return jit(optimize_epoch)
   
    nepoch = 0
    best_loss = jnp.sum(jnp.array([1e10]))
    nstep = MC_time
    eigene = jnp.sum(jnp.zeros(1))
    loss_fn = jnp.sum(jnp.zeros(1))

    opt_state = optim.init(params)
    
    print_err = print_info.Print_Info(ferr)
    
    train_epoch = outer_loop(nstep)
    Epoch = int(Epoch / nstep)

    for iepoch in range(Epoch): 
        # set the model to train
        dataloader.params = params
        coor = dataloader()
        tmp_params = params
        tmp_state = opt_state
        params, opt_state, coor, loss_train, eig_train = train_epoch(params, opt_state, coor, loss_fn, eigene)

        if jnp.isnan(loss_train):
            params = tmp_params
            opt_state = tmp_state

        nepoch = nepoch + nstep
        lr = schedule_fn(nepoch)
        loss_train = jnp.sqrt(loss_train / nstep)
        eig_train = eig_train / nstep
        print_err(nepoch, lr, eig_train, loss_train)

        if loss_train < best_loss:
            best_loss = loss_train
            save_params(params, "wf.pkl")
            params["params"]["elevel"] = params["params"]["elevel"].at[0].set(eig_train)
            np.savez("coor", coor=np.array(dataloader.seedcoor))
        sys.stdout.flush()
    

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


def make_wf_gradient(batch_le, vmap_model, sqrt_mass, clip_scale):

    def get_loss(params, mwcoor, diff, sqrt_mass):
        logpsi = vmap_model(params, mwcoor, sqrt_mass)
        return jnp.dot(diff, logpsi)

    jit_grad = jax.jit(jax.grad(get_loss))

    def _loss_and_grad(params, coor):

        local_values = jit_batch_map(params, coor)
        clip_center = jnp.median(local_values)
        variance = jnp.abs(local_values - clip_center)
        tv = jnp.mean(variance)
        clip_values = jnp.clip(local_values, clip_center - clip_scale * tv, clip_center + clip_scale * tv)
        varene = jnp.mean(clip_values)
        diff = jnp.square(clip_values - initene)
        tot_diff = diff - jnp.mean(diff)
        loss = jnp.mean(jnp.square(clip_values-varene))
       
        mwcoor = coor * sqrt_mass[None, :, None]
        grads = jit_grad(params, mwcoor, tot_diff, sqrt_mass)
        return loss, varene, grads
        

    jit_loss_and_grad = jit(_loss_and_grad)
    return jit_loss_and_grad

loss_grad_fn = make_wf_gradient(batch_le, vmap_model, sqrt_mass, clip_scale)

dataloader = dataloader.Sampling(key[1], coor, sqrt_mass, ele_ion, params, model, vmap_model, nwalker=nwalker, MC_time=MC_time, step=step)

ferr=open("pre.err","w")
ferr.write("Equivariant MPNN package based on three-body descriptors \n")
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))

nstep = MC_time

schedule_fn = optax.linear_schedule(init_value=pre_slr, end_value=pre_elr, transition_steps=pre_patience_epoch)

optim = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate = schedule_fn))

if pre_table_init==1:
    params = load_params(filename = "wf.pkl")
    dataloader.params = params

train(params, optim, loss_grad_fn, schedule_fn, dataloader, pre_Epoch)

ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
ferr.close()
print("Normal termination")
