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
    def __init__(self, key, coor, sqrt_mass, ele_ion, params, wf_model, vmap_model, nwalker=4096, MC_time=50, step = 0.2):
        self.nwalker = nwalker
        self.nele = ele_ion.shape[0]
        self.numatom = coor.shape[1]
        self.nuc = self.numatom - self.nele
        self.MC_time = MC_time
        self.sqrt_mass = sqrt_mass

        self.allatomindex=jnp.zeros(shape=(2,int(self.numatom * (self.numatom-1)/2)), dtype=jnp.int32)
        neigh=jnp.arange(self.numatom, dtype=jnp.int32)
        num=0
        for i in range(self.numatom-1):
            self.allatomindex = self.allatomindex.at[0,num:num+self.numatom-1-i].set(i )
            self.allatomindex = self.allatomindex.at[1,num:num+self.numatom-1-i].set(neigh[i+1:])
            num=num+self.numatom-1-i

        key = jax.random.split(key, 2)
        # params for electron
        step = jnp.array([step[0]]*self.nele + [step[1]] * self.nuc)
        self.step = step/jnp.square(sqrt_mass)

        try: 
            coor = np.load("coor.npz")["coor"]
            coor = jnp.array(coor)[:nwalker]
            print("load data")

        except FileNotFoundError:
            nrepeat = math.ceil(nwalker/coor.shape[0]) 
            coor = jnp.repeat(coor.reshape(1, -1, self.numatom, 3), nrepeat, axis=0).reshape(-1, self.numatom, 3)[:nwalker]
            offset = coor[:, ele_ion + self.nele]
            init_ele = jax.random.normal(key[1], (nwalker, self.nele, 3)) + offset
            coor = coor.at[:,:self.nele].set(init_ele)
            print("random initilization")
        
        
        self.seedcoor = coor

        self.param = params
        self.wf_model = jit(wf_model.apply)
        self.vmap_model = vmap_model
      
        self.key = key[-1]

        self.MC = self.MC_sample()

    def __call__(self):

        mwcoor = self.seedcoor * self.sqrt_mass[:, None]
        init_logpsi = self.vmap_model(self.params, mwcoor, self.sqrt_mass)

        in_key = jax.random.split(self.key, self.nwalker+1)

        (tmp, self.seedcoor, init_logpsi, key), store = self.MC(self.params, self.seedcoor, init_logpsi, in_key[:self.nwalker])

        self.key = in_key[-1]
        sample = jnp.concatenate(store, 0)
        return sample

    def MC_sample(self):
        @jit
        def _MC_step(params, initcoor, init_logpsi, key):
            key = jax.random.split(key, 3)
            randsize = jax.random.normal(key[0], (self.numatom, 3))
            stepsize = randsize * self.step[:, None]

            coor = initcoor + stepsize
            mwcoor = coor * self.sqrt_mass[:, None]
            logpsi = self.wf_model(params, mwcoor, self.sqrt_mass)

            expand_coor = coor[self.allatomindex.reshape(-1)].reshape(2, -1, 3)
            distvec = expand_coor[1] - expand_coor[0]
            distances = jnp.linalg.norm(distvec, axis=1)
            judge1 = jnp.greater(jnp.min(distances), 0.05)

            alpha = jax.random.uniform(key[1], 1)
            judge = (2.0 * (logpsi - init_logpsi)) > jnp.log(alpha)
            judge = jnp.logical_and(judge, judge1)[0]

            initcoor = coor * judge + initcoor * (1.0 - judge)
            init_logpsi = logpsi * judge + init_logpsi * (1.0 - judge)
            return initcoor, init_logpsi, key[2]

        jit_mc = jit(vmap(_MC_step, in_axes=(None, 0, 0, 0), out_axes=(0, 0, 0)))

        def scan_fun(carry, _):
            params, coor, logpsi, key = carry
            coor, logpsi, key = jit_mc(params, coor, logpsi, key)
            return (params, coor, logpsi, key), coor
  
        def jax_scan(params, initcoor, init_logpsi, key):
            return jax.lax.scan(scan_fun, (params, initcoor, init_logpsi, key), None, length=self.MC_time)

        return jit(jax_scan)
