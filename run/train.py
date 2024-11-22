#! /usr/bin/env python3

import sys
import numpy as np
from src.params import *
import model.MPNN_det as MPNN
import model.MPNN_pes as MPNN_pes
import dataloader.dataloader as dataloader
import src.print_info as print_info
import optax
from src.save_model import save_params, load_params
from src.hamiltonian_total import LocalEnergy
from src.print_params import print_params
from jax import vmap, jit
import folx


# train function
def train(key, params, optim, schedule_fn, value_and_grads_fn, value_fn, dataloader, Epoch):


    def train_loop(ncyc):

        def optimize_epoch(carry):

            def body(i, carry):
                params, opt_state, coor, loss_fn = carry
                loss, grads = value_and_grads_fn(params, coor[i])
                updates, opt_state = optim.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                loss_fn += loss
                return params, opt_state, coor, loss_fn

            key, count, nstep, threshod, params, opt_state, coor, loss_fn = carry   
            key = jax.random.split(key, 2)
            shuffle_list = jax.random.permutation(key[0], coor.shape[0])
            rcoor = coor[shuffle_list]   
            rcoor = rcoor.reshape(-1, batchsize, numatom, 3)
            loss = jnp.sum(jnp.zeros(1))

            params, opt_state, rcoor, loss = jax.lax.fori_loop(0, nstep, body, (params, opt_state, rcoor, loss))
            count += 1
            loss_fn = jnp.sqrt(loss/nstep)
            return key[-1], count, nstep, threshod, params, opt_state, coor, loss_fn


        def cond_fun(carry):
            key, count, nstep, threshod, params, opt_state, coor, loss_fn = carry
            return (loss_fn > threshod) & (count < (ncyc - 0.5))

        def optimize_cyc(key, nstep, params, opt_state, coor, loss_train):
            threshod = 0.9 * loss_train
            count = 0
            key, count, nstep, threshod, params, opt_state, coor, loss = jax.lax.while_loop(cond_fun, optimize_epoch, (key, count, nstep, threshod, params, opt_state, coor, loss_train))
            return params, opt_state, count

        return jax.jit(optimize_cyc)

    def sel_coor():

        def _sel(key, loss_train, div_val, coor, sample):

            ulimit = jnp.minimum(loss_train * 5, 1e-2)
            llimit = jnp.minimum(loss_train * 2, 4e-3)
            judge = jnp.logical_and(jnp.greater(div_val, llimit), jnp.less(div_val, ulimit))
            shuffle_list = jax.random.permutation(key[0], coor.shape[0]) 
            judge = judge[shuffle_list]
            index_list = jnp.nonzero(judge, size = n_add)[0]
            sel = coor[shuffle_list[index_list]]
            sample = jnp.concatenate((sample, sel), axis=0)
            shuffle_list = jax.random.permutation(key[1], sample.shape[0]) 
            return sample[shuffle_list]

        return jit(_sel)


    process_data = sel_coor() 

    opt_state = optim.init(params)
    
    print_err = print_info.Print_Info(ferr)
    train_fn = train_loop(ncyc)

    nepoch = 0
    istep = 0
    best_loss = jnp.sum(jnp.array([1e10]))

#  generate the initial coor 
    key = jax.random.split(key, 2)
    sample = dataloader.seedcoor
    train_coor = sample
    loss_train = jnp.sum(jnp.ones(1))
    nstep = jnp.floor_divide(nwalker, batchsize)
    nval = nstep

    while nepoch < Epoch: 
        key = jax.random.split(key[-1], 4)
        dataloader.params = params
        coor = dataloader()
        train_coor = jnp.concatenate((coor, train_coor), axis=0)
        

        tmp_params = params
        tmp_state = opt_state

        params, opt_state, count = train_fn(key[0], nstep+nval, params, opt_state , train_coor, loss_train)

        loss_train, eig_train, div_train = value_fn(params, train_coor)

        if jnp.isnan(loss_train):
            params = tmp_params
            opt_state = tmp_state



# print and save information
        nepoch += count 
        istep += count * (nstep + nval)
        lr = schedule_fn(istep)
 
        print_err(nepoch, lr, eig_train, loss_train)

        save_params(params, "wf.pkl")
        np.savez("coor", coor=np.array(dataloader.seedcoor))
        np.savez("sample", coor=np.array(sample))

        sample = process_data(key[1:3], loss_train, div_train[:nwalker], coor, sample)
        numpoint = sample.shape[0]
        nstep = jnp.floor_divide(numpoint, batchsize)
        nstart = numpoint - nstep * batchsize
        train_coor = sample[nstart:]


        sys.stdout.flush()

    

key = jax.random.split(key[-1], 4)

#==============================Equi MPNN==============================================================
model = MPNN.MPNN(emb_nl, MP_nl, det_nl, out_nl, numatom, nele, initene, spin_inp, neighlist, en_neighlist, nn_neighlist, index_ee, input_embed, cusp, index_l, index_i1, index_i2, ens_cg, index_add, index_cg, initbias_cg, init_contract, initbias_neigh, initbias_mp, initbias_det, initbias_embne, initbias_out, epson, sph_wf,  nn_table=nn_table, ndet=ndet, nwave=nwave, ncontract=ncontract, ngauss=ngauss, rmaxl=rmaxl, MP_loop=MP_loop)


params_rng = {"params": key[0]}
params = model.init(params_rng, coor[0], sqrt_mass)

print("NN structure for wavefunction")
print_params(params)

vmap_model = jit(vmap(model.apply, in_axes=(None, 0, None)))


Hop = LocalEnergy(numatom, charge, sqrt_mass, model, sparsity=sparsity)

batch_le = jit(vmap(Hop, in_axes=(None, 0)))

def make_gradient(batch_le, sqrt_mass, clip_scale):

    def get_loss(params, coor):

        local_values = batch_le(params, coor)
        clip_center = jnp.median(local_values)
        variance = jnp.abs(local_values - clip_center)
        tv = jnp.mean(variance)
        judge = jnp.logical_and(jnp.greater(local_values, clip_center - clip_scale * tv), jnp.less(local_values, clip_center+ clip_scale * tv))

        div = jnp.abs(local_values - params["params"]["elevel"]) * judge
        loss = jnp.sum(jnp.square(div)) / jnp.sum(judge)
        return loss
    return jax.jit(jax.value_and_grad(get_loss))

# use different train and val for saving the jit compile time
value_and_grads_fn = make_gradient(batch_le, sqrt_mass, clip_scale)

def batch_map(batch_le, sqrt_mass, clip_scale):

    def _batch_value(params, coor):
        f_closure = lambda x: batch_le(params, x)
        local_values = jax.lax.map(f_closure, coor.reshape(-1, batchsize, numatom, 3))
        return local_values

    def _batch_loss(params, coor):
        local_values = _batch_value(params, coor)
        local_values = local_values.reshape(-1)
        clip_center = jnp.median(local_values)
        variance = jnp.abs(local_values - clip_center)
        tv = jnp.mean(variance)
        judge = jnp.logical_and(jnp.greater(local_values, clip_center - clip_scale * tv), jnp.less(local_values, clip_center+ clip_scale * tv))

        div = jnp.abs(local_values - params["params"]["elevel"]) 
        loss = jnp.sqrt(jnp.sum(jnp.square(div) * judge) / jnp.sum(judge))
        varene = jnp.sum(local_values * judge) / jnp.sum(judge)
        return loss, varene, div
    return jax.jit(_batch_loss)

value_fn = batch_map(batch_le, sqrt_mass, clip_scale)       

schedule_fn = optax.linear_schedule(init_value=slr, end_value=elr, transition_steps=patience_step)

optim = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate = schedule_fn, weight_decay=1e-6))

dataloader = dataloader.Sampling(key[1], coor, sqrt_mass, ele_ion, params, model, vmap_model, nwalker=nwalker, MC_time=MC_time, step=step)


ferr=open("wf.err","w")
ferr.write("Equivariant MPNN package based on three-body descriptors \n")
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))

if table_init==1:
    params = load_params(filename = "wf.pkl")
    print("load pes and wf model")

train(key[2], params, optim, schedule_fn, value_and_grads_fn, value_fn, dataloader, Epoch)

ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
ferr.close()
print("Normal termination")
