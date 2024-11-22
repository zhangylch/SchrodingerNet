import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from functools import partial
import jax.random as jrm
import scipy

class SPH_CAL():
    '''
    This module perform the calculation of spherical harmonic expansion based on the derivation in this work (https://arxiv.org/abs/1410.1748).
    max_l: int32/int64
         represents the maximal angular quantum numebr for the evaluation of spherical harmonic. Example: 2

    Dtype: jnp.float32/jnp.float64
         represents the datatype in this module. Example: jnp.float32
    '''
    def __init__(self, max_l=1):
        if max_l<0.5: raise ValueError("The angular momentum must be greater than or equal to 1. Or the angular momentum is lack of angular information, the calculation of the sph is meanless.")
        self.max_l = max_l + 1
        self.pt = jnp.arange(self.max_l * self.max_l).reshape(self.max_l,self.max_l)
        self.yr = jnp.arange(self.max_l * self.max_l).reshape(self.max_l,self.max_l)
        self.yr_rev = jnp.arange(self.max_l * self.max_l).reshape(self.max_l,self.max_l)
        num_lm = int((self.max_l+1) * self.max_l/2)
        coeff_a = jnp.zeros(num_lm)
        coeff_b = jnp.zeros(num_lm)
        tmp = jnp.arange(self.max_l)
        self.prefactor1 = -jnp.sqrt(1.0+0.5/tmp)
        self.prefactor1 = self.prefactor1.at[0].set(0.0)
        self.prefactor2 = jnp.sqrt(2.0*tmp+3)
        ls = tmp*tmp
        for l in range(self.max_l):
            self.pt = self.pt.at[l, 0:l+1].set(tmp[0:l+1] + int(l*(l+1)/2))
            # here the self.yr and self.yr_rev have overlap in m=0.
            self.yr = self.yr.at[l, 0:l+1].set(ls[l] + l + tmp[0:l+1])
            self.yr_rev = self.yr_rev.at[l, 0:l+1].set(ls[l] + l - tmp[0:l+1])
            if l>0.5:
                coeff_a = coeff_a.at[self.pt[l, 0:l]].set(jnp.sqrt((4.0 * ls[l] - 1) / (ls[l] - ls[0:l])))
                coeff_b = coeff_b.at[self.pt[l, 0:l]].set(-jnp.sqrt((ls[l-1] - ls[0:l]) / (4.0*ls[l-1] - 1.0)))

        self.sqrt2_rev = jnp.sqrt(1.0 / 2.0)
        self.sqrt2pi_rev = jnp.sqrt(0.5 / np.pi)
        self.hc_factor1 = jnp.sqrt(15.0 / 4.0 / np.pi)
        self.hc_factor2 = jnp.sqrt(5.0 / 16.0 / np.pi)
        self.hc_factor3 = jnp.sqrt(15.0 / 16.0 / np.pi)
        self.coeff_a = coeff_a
        self.coeff_b = coeff_b


    @partial(jit, static_argnums=0)
    def __call__(self,cart):
        '''
        cart: jnp.float32/jnp.float64.
            represents the cartesian coordinates of systems with its shape [3,...]. 
        '''
        distances = jnp.linalg.norm(cart, axis=0)  # to convert to the dimension (n,batchsize)
        d_sq = distances * distances
        sph_shape = (self.max_l * self.max_l,) + cart.shape[1:]
        sph = jnp.zeros(sph_shape)
        sph = sph.at[0].set(self.sqrt2pi_rev * self.sqrt2_rev)
        sph = sph.at[1].set(self.prefactor1[1] * self.sqrt2pi_rev * cart[1])
        sph = sph.at[2].set(self.prefactor2[0] * self.sqrt2_rev * self.sqrt2pi_rev * cart[2])
        sph = sph.at[3].set(self.prefactor1[1] * self.sqrt2pi_rev * cart[0])
        if self.max_l>2.5:
            sph = sph.at[4].set(self.hc_factor1 * cart[0] * cart[1])
            sph = sph.at[5].set(-self.hc_factor1 * cart[1] * cart[2])
            sph = sph.at[6].set(self.hc_factor2 * (3.0 * cart[2] * cart[2] - d_sq))
            sph = sph.at[7].set(-self.hc_factor1 * cart[0] * cart[2])
            sph = sph.at[8].set(self.hc_factor3 * (cart[0] * cart[0] - cart[1] * cart[1]))
            for l in range(3, self.max_l):
                sph = sph.at[self.yr[l, 0:l-1]].set(jnp.einsum("i, i...-> i...",self.coeff_a[self.pt[l, 0:l-1]], (cart[2] * sph[self.yr[l-1, 0:l-1]] + jnp.einsum("i, ..., i... -> i...", self.coeff_b[self.pt[l, 0:l-1]], d_sq, sph[self.yr[l-2, 0:l-1]]))))
                sph = sph.at[self.yr_rev[l, 1:l-1]].set(jnp.einsum("i, i... -> i...", self.coeff_a[self.pt[l, 1:l-1]], (cart[2] * sph[self.yr_rev[l-1, 1:l-1]] + jnp.einsum("i,..., i... -> i...", self.coeff_b[self.pt[l, 1:l-1]], d_sq, sph[self.yr_rev[l-2, 1:l-1]]))))
                sph = sph.at[self.yr[l, l-1]].set(self.prefactor2[l-1] * cart[2] * sph[self.yr[l-1, l-1]])
                sph = sph.at[self.yr_rev[l, l-1]].set(self.prefactor2[l-1] * cart[2] * sph[self.yr_rev[l-1, l-1]])
                sph = sph.at[self.yr[l, l]].set(self.prefactor1[l] * (cart[0] * sph[self.yr[l-1, l-1]] - cart[1] * sph[self.yr_rev[l-1, l-1]]))
                sph = sph.at[self.yr_rev[l, l]].set(self.prefactor1[l] * (cart[0] * sph[self.yr_rev[l-1, l-1]] + cart[1] * sph[self.yr[l-1, l-1]]))
        return sph
'''
# here is an example to use the sph calculation
import wigners
max_l=7
key=jrm.PRNGKey(0)
init_key=jrm.split(key)
cart=jrm.uniform(key,(3,10))
distances=jnp.linalg.norm(cart,axis=0)
cart=cart/distances
rotate=jnp.zeros((3,3))
ceta =np.pi*60.0/180.0
rotate=rotate.at[0,0].set(jnp.cos(ceta))
rotate=rotate.at[1,1].set(jnp.cos(ceta))
rotate=rotate.at[0,1].set(jnp.sin(ceta))
rotate=rotate.at[1,0].set(-jnp.sin(ceta))
rotate=rotate.at[2,2].set(1.0)
cart1=jnp.einsum("ij,jk->ik",rotate,cart)
sph=SPH_CAL(max_l=max_l)

def contraction(cart,j1,j2,j3):
    sph_complex=sph(cart)
    sum_sph=jnp.sum(sph_complex,axis=1)
    feature=0
    num1=j1*j1
    num2=j2*j2
    num3=j3*j3
    for m1 in range(-j1,j1+1):
        m2_down=max(-j2,-j3-m1)
        m2_up=min(j2+1,j3-m1+1)
        for m2 in range(m2_down,m2_up):
            m3=-m1-m2
            feature+=sum_sph[num1+j1+m1]*sum_sph[num2+j2+m2]*sum_sph[num3+j3+m3]*wigners.wigner_3j(j1,j2,j3,m1,m2,m3)
    return feature

print(contraction(cart,3,2,6))
print(contraction(cart1,3,2,6))

#print(jax.make_jaxpr(sph.compute_sph)(cart))
jax.lax.stop_gradient(cart)
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp=sph(cart)       
print("The time difference is :", timeit.default_timer() - starttime)
print(tmp)
print(cart.reshape(-1))
forward=jax.jit(jax.vmap(test_forward,in_axes=(1),out_axes=(1)))
forward(cart)
starttime = timeit.default_timer()
print("The start time is :",starttime)
tmp=forward(cart)
print("The time difference is :", timeit.default_timer() - starttime)
print(tmp.shape)

#jac=jax.jit(jax.vmap(jax.jacfwd(test_forward),in_axes=(1),out_axes=(1)))
#grad=jac(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#grad=jac(cart)
#print("The time difference is :", timeit.default_timer() - starttime)

#hess=jax.jit(jax.vmap(jax.hessian(sph.compute_sph),in_axes=(1),out_axes=(1)))
#tmp=hess(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=hess(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#
## calculate hessian by jac(jac)
#hess=jax.jit(jax.vmap(jax.jacfwd(jax.jacfwd(sph.compute_sph)),in_axes=(1),out_axes=(1)))
#tmp=hess(cart)
#starttime = timeit.default_timer()
#print("The start time is :",starttime)
#tmp=hess(cart)
#print("The time difference is :", timeit.default_timer() - starttime)
#print(tmp.shape)
'''
