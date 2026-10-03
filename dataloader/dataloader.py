import numpy as np
import jax 
import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
from functools import partial
import math


# all in atomic unit
# Temperature 1 au = 3.1577e5 K
# time 1 au = 2.41888e-17 s 
# constant


class Sampling():
    def __init__(self, sqrt_mass, model, MC_time=50, jnp_dtype="float32", nwalker=4096, step=0.2, prob=0.5, nmove=4):

        self.nele = int(jnp.sum(jnp.less(sqrt_mass, 2)))
        self.npar = sqrt_mass.shape[0]
        self.nuc = self.npar - self.nele
        self.MC_time = MC_time
        self.sqrt_mass = sqrt_mass
        self.prob=prob
        self.nwalker=nwalker
        self.nmove = min(self.nele, nmove)

        if "32" in jnp_dtype:
            np_int = np.int32
        else:
            np_int = np.int64

        allatomindex=np.zeros(shape=(2,int(self.npar * (self.npar-1)/2)), dtype=np_int)
        neigh=np.arange(self.npar, dtype=np_int)
        num=0
        for i in range(self.npar-1):
            allatomindex[0,num:num+self.npar-1-i] = i 
            allatomindex[1,num:num+self.npar-1-i] = neigh[i+1:]
            num=num+self.npar-1-i
        self.allatomindex = jnp.array(allatomindex)
        # params for electron
        self.wf_model = model

        self.vmap_model = vmap(model, in_axes=(None, 0))
        step = jnp.array([step[0]]*self.nele + [step[1]] * self.nuc, dtype=sqrt_mass.dtype)
        self.step = step/jnp.square(sqrt_mass) * 1e-1
      
        self.MC = self.MC_sample()

    def __call__(self, key, coor, init_logpsi, params, step, scale=1.0):

        coor = jnp.asarray(coor, dtype=self.sqrt_mass.dtype)
        step = jnp.asarray(step, dtype=coor.dtype)
        mwcoor = coor * self.sqrt_mass[:, None]
        init_logpsi = self.vmap_model(params, mwcoor)
        scale = jnp.asarray(scale, dtype=init_logpsi.dtype)
        params, seedcoor, init_logpsi, key, step = self.MC(params, coor, init_logpsi, key, step, scale)
        

        return key, seedcoor, init_logpsi, step

    def MC_sample(self):
        def _MC_step(params, initcoor, init_logpsi, key, step, scale):
            key = jax.random.split(key, 4)
            randsize = jax.random.normal(key[0], (self.nmove+self.nuc, 3), dtype=initcoor.dtype)
            stepsize = randsize * step[-self.nmove-self.nuc:, None]
            move_idx = jax.random.choice(key[1], self.nele, (self.nmove,), replace=False)

            coor = initcoor.at[move_idx].add(stepsize[:self.nmove])
            coor = coor.at[self.nele:].add(stepsize[self.nmove:])
            mwcoor = coor * self.sqrt_mass[:, None]
            logpsi = self.wf_model(params, mwcoor)

            alpha = jax.random.uniform(key[2], (1,), dtype=init_logpsi.dtype)
            judge = jnp.greater(2.0 * (logpsi - init_logpsi) / scale, jnp.log(alpha))
            judge = jnp.logical_and(judge, jnp.isfinite(logpsi))
            judge = judge.astype(initcoor.dtype)

            initcoor = jnp.where(judge[0], coor, initcoor)
            init_logpsi = jnp.where(judge[0], logpsi, init_logpsi)
            return initcoor, init_logpsi, judge, key[3]

        vmap_mc = vmap(_MC_step, in_axes=(None, 0, 0, 0, None, None), out_axes=(0, 0, 0, 0))

        def scan_fun(_, carry):
            params, coor, logpsi, key, step, scale, sum_move = carry
            coor, logpsi, judge, key = vmap_mc(params, coor, logpsi, key, step, scale)
            sum_move += jnp.sum(judge)
            return params, coor, logpsi, key, step, scale, sum_move
  
        def jax_scan(params, initcoor, init_logpsi, key, step, scale):
            sum_move = jnp.array(0.0, dtype=initcoor.dtype)
            params, initcoor, init_logpsi, key, step, scale, sum_move = jax.lax.fori_loop(0, self.MC_time, scan_fun, (params, initcoor, init_logpsi, key, step, scale, sum_move))
            step = jax.lax.psum(sum_move, axis_name="GPUs") / self.nwalker / self.MC_time / self.prob * step 
            step = jnp.where(step > self.step, step, self.step).astype(initcoor.dtype)
            return params, initcoor, init_logpsi, key, step

        return jax_scan
