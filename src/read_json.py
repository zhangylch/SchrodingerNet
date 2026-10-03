import os
from dataclasses import dataclass, field, replace
from typing import List, Any
import json

@dataclass
class JsonConfig:
    #======general setup===========================================
    local_size: int = field(default=1)
    seed: int = field(default=20)
    sparsity: int = field(default=0)
    ineig: float = field(default=1.0)
    inloss: float = field(default=1.0)
    ele_ion: List[float] = field(default_factory = lambda: [0, 1])
    spin_orb: List[float] = field(default_factory = lambda: [0.5, -0.5])
    eee: bool = field(default=False)
    nn_table: bool = field(default=True)
    neee: int = field(default=1) # batchsize for each process
    restart: bool = field(default=False)        
    jnp_dtype: str = field(default='float32')   #float32/float64
    batchsize: int = field(default=4096) # batchsize for each process
    sbatchsize: int = field(default=128) # batchsize for each process
    ckpath: str = field(default=os.getcwd()+"/ckpt_dir/")
    datapath: str = field(default="./")
    init_weight: float = field(default=0.0)
    final_weight: float = field(default=10.0)

    #  for train using local_energy
    spin_table: int = field(default=0)        
    prob: float = field(default=0.5)
    nmove: int = field(default=4)
    clip_norm: float = field(default=2.5)
    scale_nuc: float = field(default=1.0)
    scale_ele: float = field(default=1.0)
    scale_nn: float = field(default=1.0)
    scale_en: float = field(default=8.0)
    scale_ee: float = field(default=8.0)
    weight_hj: float = field(default=0.0)
    clip_scale: float = field(default=10.0)
    ncyc: int = field(default=50)
    ninit_step: int = field(default=500)
    nadd: int = field(default=16)
    ndel: int = field(default=32)
    Epoch: int = field(default=20000)                    # total numbers of epochs for fitting 
    init_lr: float = field(default=1e-1) # initial learning rate
    slr: float = field(default=1e-1) # initial learning rate
    elr: float = field(default=1e-5)                    # final learning rate
    patience_step: int = field(default=50)    
    begin_step: int = field(default=200)

    
    nwalker: int = field(default=4096)
    MC_time: int = field(default=20)
    step: List[float] = field(default_factory = lambda: [0.1, 1.0])
    
    #===============================embedded NN structure==========
    rden: float=field(default=4.0)
    en_eps: float=field(default=0.02)
    ee_eps: float=field(default=0.02)
    nn_eps_small: float=field(default=0.1)
    nn_eps_large: float=field(default=0.1)
    where_eps: float=field(default=1e-10)
    maxl: int = field(default=3)
    pmaxl: int = field(default=2)
    orbl: int = field(default=0)
    eorbl: int = field(default=0)
    nwave: int = field(default=16)
    nbasis: int = field(default=32)
    ndet: int = field(default=1)
    ngauss: int = field(default=64)
    nradial: int = field(default=64)
    MP_loop: int = field(default=3)
    
    use_norm: bool = field(default=False)
    radial_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, False])
    den_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, False])
    MP_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, False])
    alpha_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, True]) 
    alphaee_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, True]) 
    orb_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, False]) 
    lorb_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, False]) 
    coeff_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, False]) 
    coeffee_nl: List[Any] = field(default_factory = lambda: [1, 64, 2, False, False]) 
    emb_nl: List[Any] = field(default_factory = lambda: [2, 128, 2, False, False]) 
    out_nl: List[Any] = field(default_factory = lambda: [2, 256, 2, False, True]) 
    nuc_nl: List[Any] = field(default_factory = lambda: [2, 256, 2, False, False]) 
    #====================VMC pretrain==================
    vmc_clip_scale: float = field(default=5.0)
    vmc_clip_norm: float = field(default=2.5)
    vmc_restart: bool = field(default=False)
    vmc_Epoch: int = field(default=20000)                    # total numbers of epochs for fitting 
    vmc_init_lr: float = field(default=0.0001)
    vmc_slr: float = field(default=0.005)
    vmc_elr: float = field(default=5e-5)
    vmc_patience_step: int = field(default=200)
    vmc_begin_step: int = field(default=200)
    vmc_ncyc: int = field(default=10)
    vmc_MC_time: int = field(default=5)
    vmc_numpoint: int = field(default=5)
    vmc_step: List[float] = field(default_factory = lambda: [0.1, 1.0])
    


def load_config(json_path: str) -> JsonConfig:
    default_config = JsonConfig()  # 默认配置
    
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        return default_config
    
    # 仅更新 JSON 中提供的字段
    return replace(default_config, **json_data)

