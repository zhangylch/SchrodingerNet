import time
import numpy as np
import os
import sys
import jax 
import jax.numpy as jnp
import dataloader.read_data as read_data
from low_level import sph_cal, cg_cal
from src.gpu_sel import gpu_sel

gpu_sel()

#======general setup===========================================
table_init=0                   # 1: a pretrained or restart  
dtype='float32'   #float32/float64
initene = -1.5
# batchsize: the most import setup for efficiency
batchsize = 2048  # batchsize for each process
numatom = 4
ele_ion = [0, 1]
sparsity = 0

#======= electron sample============a
step = 0.2
MC_time = 5
#======= nuclear sample============
N_temp=1e-4
N_timestep=10.0
N_scaltime=2500

#========================parameters for optim=======================
Epoch=50000                    # total numbers of epochs for fitting 
re_epoch = 5
slr=0.0005                  # initial learning rate
elr=1e-5                  # final learning rate
patience_epoch = 20000             # patience epoch  Number of epochs with no improvement after which learning rate will be reduced. 

datafloder="./"

#=======================parameters for local environment========================
ndet = 32
nele = 2
nup =1
cutoff = 8.0
max_l=1
nwave=8
ngauss=8
ncontract=8
epson = 1e-12

#===============================embedded NN structure==========
emb_nl = [1, 128, 1, False]  # nblock, nfeature, nlayer, Layer_norm

MP_loop = 2
MP_nl = [1, 128, 2, True]  # nblock, nfeature, nlayer, Layer_norm

det_nl = [1, 128, 2, True]  # nblock, nfeature, nlayer, Layer_norm

out_nl = [2, 128, 2, True]  # nblock, nfeature, nlayer, Layer_norm

#======================read input=================================================================
with open('para/input.in') as f1:
   while True:
      tmp=f1.readline()
      if not tmp: break
      string=tmp.strip()
      if len(string)!=0:
          if string[0]=='#':
              pass
          else:
              m=string.split('#')
              exec(m[0])


if dtype=='float64':
    jax.config.update("jax_enable_x64", True)

ndown = nele-nup
rmaxl = max_l + 1

def contract_sph(rmaxl, MP_loop):
    index_i1 = []
    index_i2 = []
    cg_array = []
    index_add = []
    index_cg = []
    num_coeff = 0
    initbias_cg = []
    for lf in range(rmaxl):
        for li1 in range(rmaxl):
            low = abs(li1 - lf)
            up = min(rmaxl, li1+lf+1)
            for li2 in range(low, up):
                num_coeff +=1
                if np.mod(li1+li2, 2) == 0: 
                    initbias_cg.append(np.random.normal())
                else:
                    initbias_cg.append(np.random.normal() * 10.0)
    
                cg = cg_cal.clebsch_gordan(li1, li2, lf)
                for mf in range(0, 2*lf+1):
                    for mi1 in range(0, 2*li1+1):
                        for mi2 in range(0, 2*li2+1):
                            dim2 = li2 * li2 +  mi2 
                            dim1 = li1 * li1 +  mi1
                            dim3 = lf * lf + mf
                            if np.abs(cg[mi1, mi2, mf]) > 1e-3:
                                index_i1.append(dim1)
                                index_i2.append(dim2)
                                index_cg.append(num_coeff)
                                cg_array.append(cg[mi1, mi2, mf])
                                index_add.append(dim3)
    index_i1 = jnp.asarray(index_i1)
    index_i2 = jnp.asarray(index_i2)
    ens_cg = jnp.asarray(cg_array)
    index_add = jnp.asarray(index_add)
    index_cg = jnp.asarray(index_cg)
    initbias_cg = jnp.asarray(initbias_cg).reshape(1,-1).repeat(MP_loop+1, axis=0).reshape(-1)

    index_l = jnp.arange(rmaxl*rmaxl)
    for l in range(rmaxl):
        index_l = index_l.at[l*l:(l+1)*(l+1)].set(l)

    return index_i1, index_i2, ens_cg, index_add, index_cg, initbias_cg, index_l


key = jax.random.PRNGKey(12)

ele_ion = jnp.array(ele_ion)

coor, pot, mass, charge, species = read_data.Read_data(datafloder=datafloder)
sqrt_mass = jnp.sqrt(mass)

numatom = species.shape[0]
neighlist = jnp.arange(2 * int(numatom * (numatom - 1))).reshape(2, int(numatom * (numatom - 1)))
num = 0
for i in range(numatom):
    for j in range(numatom):
        if abs(i - j) > 0.5:
             neighlist = neighlist.at[0, num].set(i)
             neighlist = neighlist.at[1, num].set(j)
             num += 1

judge = jnp.logical_and(jnp.less(neighlist[0], nele - 0.5), \
                        jnp.greater(neighlist[1], nele-0.5))

en_neighlist = jnp.nonzero(judge)[0]
nn_neighlist = en_neighlist
if numatom - nele > 1.5:
    judge = jnp.logical_and(jnp.greater(neighlist[0], nele - 0.5), \
                            jnp.greater(neighlist[1], nele-0.5))
    nn_neighlist = jnp.nonzero(judge)[0]

index_ee = en_neighlist
if nele > 1.5:
    judge = jnp.logical_and(jnp.less(neighlist[0], nele-0.5), \
                           jnp.less(neighlist[1], nele-0.5))
    index_ee = jnp.nonzero(judge)[0]



index_i1, index_i2, ens_cg, index_add, index_cg, initbias_cg, index_l = \
    contract_sph(rmaxl, MP_loop)

neigh_spec = species[neighlist[0]]
center_spec = species[neighlist[1]]
add_spec = neigh_spec + center_spec
mul_spec = neigh_spec * center_spec
input_embed = jnp.concatenate((add_spec, mul_spec), axis=1)
spin_inp = (species[:nele, None] + species[None, :nele]).reshape(-1,1)

cusp = -jnp.ones(neighlist.shape[1]).reshape(-1) * jnp.abs(mul_spec).reshape(-1) * 2

if nele > 1.5:
    coeff = jnp.abs(neigh_spec[index_ee] + center_spec[index_ee]) + 1.0
    tmp = -cusp[index_ee] / coeff.reshape(-1)
    cusp = cusp.at[index_ee].set(tmp)


nn_table = False
if numatom - nele > 1.5:
    nn_table = True
    cusp = cusp.at[nn_neighlist].set(0.0)
cusp = cusp / 2.0
print(cusp)

key = jax.random.split(key, 9)
delta = nwave/cutoff
alpha = jax.random.uniform(key[0], nwave)*delta + delta/2.0
rs = jnp.arange(nwave) * cutoff / nwave
initbias = jax.random.uniform(key[1], shape=(ncontract * rmaxl,))
initbias_neigh = jnp.concatenate((alpha, rs, initbias))

init_contract = jax.random.normal(key[2], shape=(MP_loop+2, rmaxl, ncontract, nwave)) / nwave
initbias_mp = jax.random.normal(key[3], shape=(ncontract * rmaxl,)) / (numatom - 1)
initbias_det = jax.random.normal(key[4], shape=(nele * ndet,))
initbias_embne = jax.random.normal(key[5], shape=(nele * ndet * 2,))
alpha1 = jax.random.uniform(key[6], shape=(ngauss,)) * 0.5 + 0.25
alpha2 = jax.random.uniform(key[7], shape=(ngauss*2,))
initbias_out = jnp.concatenate((alpha1, alpha2))

# define the class for the calculation of spherical harmonic expansion
sph_wf=sph_cal.SPH_CAL(max_l=max_l)
initene = jnp.array([initene])

