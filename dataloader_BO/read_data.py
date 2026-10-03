import numpy as np
import jax.numpy as jnp

# read system configuration and energy
def Read_data(datafloder="./"):
    fname2=datafloder+'configuration'
    coor=[]
    pot = []
    with open(fname2,'r') as f1:
        while True:
            string=f1.readline()
            if not string: break
            tmp = string.split()
            numatom = int(tmp[0])
            pot.append(float(tmp[1]))
            # here to save the coordinate with row first to match the neighluist in fortran
            icoor=np.zeros((numatom,3))
            mass=np.zeros(numatom)
            charge=np.zeros(numatom)
            species=np.zeros((numatom,1))
            for num in range(numatom):
                string=f1.readline()
                m=string.split()
                tmp=np.array(list(map(float,m[1:])))
                mass[num]=tmp[0]
                charge[num]=tmp[1]
                species[num,0]=tmp[2]
                icoor[num]=tmp[3:6]
            coor.append(icoor)
    coor = np.array(coor)  
    return jnp.array(coor), jnp.array(pot), jnp.array(mass), jnp.array(charge), jnp.array(species)
