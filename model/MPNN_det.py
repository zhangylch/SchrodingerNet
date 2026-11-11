import sys
import numpy as np
import jax
import flax 
import jax.numpy as jnp
import jax.random as jrm
from jax.numpy import dtype, array
import flax.linen as nn
from low_level import MLP, sph_cal, safe_sinc
from flax.core import freeze
# from jax import lax # 不需要 lax.scan，使用 Python 循环
from src.data_config import ModelConfig


class MPNN(nn.Module):

    config: ModelConfig

    def setup(self):
        
        dtype = self.config.initbias_neigh.dtype

        self.sph_cal = sph_cal.SPH_CAL(max_l=self.config.rmaxl-1, dtype=dtype)

        self.spec_coeff = self.param('spec_coeff', nn.initializers.normal(1.0), (self.config.nspec, self.config.nwave, self.config.nwave), dtype)

        self.contract_coeff = self.param('contract_coeff', nn.initializers.normal(1.0), (self.config.MP_loop, self.config.nspec, self.config.prmaxl*2+self.config.rmaxl, self.config.nwave, self.config.nwave), dtype)

        self.l_coeff = self.param('l_coeff', nn.initializers.normal(1.0), (self.config.MP_loop, self.config.nspec, self.config.num_cg, self.config.nwave), dtype)

        self.scale_coeff = self.param('scale_coeff', nn.initializers.normal(1.0), (self.config.ncom_spec, self.config.nradial, self.config.nradial), dtype)

        self.hl_coeff = self.param('hl_coeff', nn.initializers.normal(0.00001), (self.config.rorbl*self.config.rorbl), dtype)

        self.exp_coeff = self.param('exp_coeff', lambda rng, _: self.config.initbias_neigh, dtype)

        self.emb_coeff = self.param('emb_coeff', nn.initializers.normal(1.0), (self.config.ncom_spec, self.config.nradial), dtype)

        self.sph_coeff = self.param('sph_coeff', nn.initializers.normal(0.1), (self.config.ncom_spec), dtype)

        self.emb_spin = self.param('emb_spin', lambda rng, _: jnp.zeros((2, self.config.ndet), dtype=dtype), dtype)
         
        self.orb_emb = self.param('orb_emb', nn.initializers.normal(1.0), (self.config.nele, self.config.ndet, self.config.nspec, self.config.rorbl*self.config.rorbl, self.config.nwave), dtype)

        self.radialnn = MLP.MLP(num_output=(self.config.prmaxl+1) * self.config.nwave, num_blocks=self.config.radial_nl[0], features=self.config.radial_nl[1], layers_per_block=self.config.radial_nl[2], use_linear=self.config.radial_nl[3], use_bias=self.config.radial_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype)

        self.MPNN_list = [MLP.MLP(num_output=(self.config.prmaxl+self.config.rmaxl)*self.config.nwave, num_blocks=self.config.MP_nl[0], features=self.config.MP_nl[1], layers_per_block=self.config.MP_nl[2], use_linear=self.config.MP_nl[3], use_bias=self.config.MP_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype) for iMP_loop in range(self.config.MP_loop)]

        self.norbcoeffnn = [MLP.MLP(num_output=(2*l+1)*self.config.nbasis, num_blocks=self.config.orb_nl[0], features=self.config.orb_nl[1], layers_per_block=self.config.orb_nl[2], use_linear=self.config.orb_nl[3], use_bias=self.config.orb_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype) for l in range(self.config.rorbl)]

        OrbCoeffVmapped = nn.vmap(
            MLP.MLP,
            in_axes=None,
            out_axes=0,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            axis_size=self.config.rorbl*self.config.rorbl*self.config.nele*self.config.ndet,
        )

        self.eeorbcoeffnn = OrbCoeffVmapped(
            num_output=self.config.nbasis,
            num_blocks=self.config.emb_nl[0],
            features=self.config.emb_nl[1],
            layers_per_block=self.config.emb_nl[2],
            use_linear=self.config.emb_nl[3],
            use_bias=self.config.emb_nl[4],
            bias_init_value=None,
            cst=self.config.cst,
            name='eeorb_coeff',
            dtype=dtype
        )

        self.initnn_en = MLP.MLP(num_output=self.config.nwave, num_blocks=self.config.radial_nl[0], features=self.config.radial_nl[1], layers_per_block=self.config.radial_nl[2], use_linear=self.config.radial_nl[3], use_bias=self.config.radial_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype)

        self.radialnn_en = [MLP.MLP(num_output=2 * self.config.prmaxl * self.config.nwave, num_blocks=self.config.den_nl[0], features=self.config.den_nl[1], layers_per_block=self.config.den_nl[2], use_linear=self.config.den_nl[3], use_bias=self.config.den_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype) for iMP_loop in range(self.config.MP_loop+1)]

        self.nodenn = [MLP.MLP(num_output=self.config.rmaxl * self.config.nwave, num_blocks=self.config.node_nl[0], features=self.config.node_nl[1], layers_per_block=self.config.node_nl[2], use_linear=self.config.node_nl[3], use_bias=self.config.node_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype) for iMP_loop in range(self.config.MP_loop)]

        if self.config.eee:
            self.contract_en = self.param('contract_en', nn.initializers.normal(1.0), (2, self.config.rmaxl, self.config.nbasis, self.config.nbasis), dtype)

            self.contract_ee = self.param('contract_ee', nn.initializers.normal(1.0), (self.config.neee, 2, self.config.prmaxl, self.config.nwave, self.config.nwave), dtype)

            self.contract_sph = self.param('contract_sph', nn.initializers.normal(1.0), (2, self.config.rmaxl, self.config.neee+1, self.config.nwave, self.config.nwave), dtype)

            self.l_orbcoeffnn = MLP.MLP(num_output=self.config.num_cg_p, num_blocks=self.config.lorb_nl[0], features=self.config.lorb_nl[1], layers_per_block=self.config.lorb_nl[2], use_linear=self.config.lorb_nl[3], use_bias=self.config.lorb_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype)

            self.alpha_en = MLP.MLP(num_output=self.config.rmaxl * self.config.nbasis, num_blocks=self.config.alpha_nl[0], features=self.config.alpha_nl[1], layers_per_block=self.config.alpha_nl[2], use_linear=self.config.alpha_nl[3], use_bias=self.config.alpha_nl[4], bias_init_value=None, cst=self.config.cst, ls=self.config.scale_en, dtype=dtype)

            self.orbitalnn_en = MLP.MLP(num_output=self.config.rmaxl * self.config.nbasis, num_blocks=self.config.coeff_nl[0], features=self.config.coeff_nl[1], layers_per_block=self.config.coeff_nl[2], use_linear=self.config.coeff_nl[3], use_bias=self.config.coeff_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype)

            self.contract_basis = self.param('contract_basis', nn.initializers.normal(1.0), (2, self.config.rmaxl, (self.config.neee+1)*self.config.nwave, self.config.nbasis), dtype)

            self.fixed_ee = self.param('fixed_ee', nn.initializers.normal(0.001), (self.config.ncom_spec, self.config.neee+1, self.config.rmaxl, self.config.nwave), dtype)
           
            self.e_lcoeff = self.param('e_lcoeff', nn.initializers.normal(1.0), (self.config.neee, 2, self.config.num_cg, self.config.nwave), dtype)

            self.alpha_ee = [MLP.MLP(num_output=self.config.rmaxl * self.config.nwave, num_blocks=self.config.alphaee_nl[0], features=self.config.alphaee_nl[1], layers_per_block=self.config.alphaee_nl[2], use_linear=self.config.alphaee_nl[3], use_bias=self.config.alphaee_nl[4], bias_init_value=None, cst=self.config.cst, ls=self.config.scale_ee, dtype=dtype) for ieee in range(self.config.neee+1)]

            self.orbitalnn_sph = [MLP.MLP(num_output=self.config.rmaxl * self.config.nwave, num_blocks=self.config.coeffee_nl[0], features=self.config.coeffee_nl[1], layers_per_block=self.config.coeffee_nl[2], use_linear=self.config.coeffee_nl[3], use_bias=True, bias_init_value=None, cst=self.config.cst, dtype=dtype) for ieee in range(self.config.neee+1)]

            self.orbitalnn_ee = [MLP.MLP(num_output=self.config.prmaxl * self.config.nwave, num_blocks=self.config.coeffee_nl[0], features=self.config.coeffee_nl[1], layers_per_block=self.config.coeffee_nl[2], use_linear=self.config.coeffee_nl[3], use_bias=self.config.coeffee_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype) for ieee in range(self.config.neee)]

            self.nodenn_e = [MLP.MLP(num_output=self.config.rmaxl * self.config.nwave, num_blocks=self.config.node_nl[0], features=self.config.node_nl[1], layers_per_block=self.config.node_nl[2], use_linear=self.config.node_nl[3], use_bias=self.config.node_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype) for iMP_loop in range(self.config.neee+1)]

        else:
            self.orbitalnn_en = MLP.MLP(num_output=self.config.rorbl * self.config.nbasis, num_blocks=self.config.coeff_nl[0], features=self.config.coeff_nl[1], layers_per_block=self.config.coeff_nl[2], use_linear=self.config.coeff_nl[3], use_bias=self.config.coeff_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype)
        
            self.alpha_en = MLP.MLP(num_output=self.config.rorbl * self.config.nbasis, num_blocks=self.config.alpha_nl[0], features=self.config.alpha_nl[1], layers_per_block=self.config.alpha_nl[2], use_linear=self.config.alpha_nl[3], use_bias=self.config.alpha_nl[4], bias_init_value=None, cst=self.config.cst, ls=self.config.scale_en, dtype=dtype)


        if self.config.nn_table:
            self.outnn_alpha = MLP.MLP(num_output=2*self.config.ngauss, num_blocks=self.config.out_nl[0], features=self.config.out_nl[1], layers_per_block=self.config.out_nl[2], use_linear=self.config.out_nl[3], use_bias=self.config.out_nl[4], bias_init_value=self.config.initbias_out, cst=self.config.cst, ls=self.config.scale_nn, dtype=dtype)

            self.outnn_coeff = MLP.MLP(num_output=1, num_blocks=self.config.nuc_nl[0], features=self.config.nuc_nl[1], layers_per_block=self.config.nuc_nl[2], use_linear=self.config.nuc_nl[3], use_bias=self.config.nuc_nl[4], bias_init_value=None, cst=self.config.cst, dtype=dtype)

    def __call__(self, cart):
        dtype = cart.dtype
        det_dtype = jnp.float64 if jax.config.x64_enabled else dtype
        rmaxl_i, prmaxl_i = self.config.rmaxl, self.config.prmaxl
        rorbl_i = self.config.rorbl
        reorbl_i = self.config.reorbl
        nrealorb_i = self.config.nrealorb
        norb_i = rmaxl_i*rmaxl_i
        pnorb_i = prmaxl_i*prmaxl_i
        nrorb_i = rorbl_i * rorbl_i
        nreorb_i = reorbl_i * reorbl_i
        nwave_i, nbasis_i, nradial_i, ndet_i = self.config.nwave, self.config.nbasis, self.config.nradial, self.config.ndet
        npar_i = self.config.npar
        nele_i = self.config.nele
        nup_i = self.config.nup
        neee_i = self.config.neee
        MP_loop_i = self.config.MP_loop
        numatom_i = npar_i - nele_i
        numneigh_i = npar_i * npar_i
        ngauss_i = self.config.ngauss
        ngauss_f = jnp.array(self.config.ngauss, dtype=dtype)
        neee_f = jnp.array(neee_i, dtype=dtype)

        hl_f = jnp.array(self.config.weight_hj, dtype=dtype)
        nrorb_f = jnp.array(nrorb_i, dtype=dtype)
        prmaxl_f = jnp.array(prmaxl_i, dtype=dtype)
        rmaxl_f = jnp.array(rmaxl_i, dtype=dtype)
        nwave_f = jnp.array(nwave_i, dtype=dtype)
        nbasis_f = jnp.array(nbasis_i, dtype=dtype)
        nradial_f = jnp.array(nradial_i, dtype=dtype)
        npar_f = jnp.array(npar_i, dtype=dtype)
        numatom_f = jnp.array(numatom_i, dtype=dtype)
        nele_f = jnp.array(nele_i, dtype=dtype)
        en_eps = jnp.array(self.config.en_eps, dtype=dtype)
        ee_eps = jnp.array(self.config.ee_eps, dtype=dtype)
        nn_eps_small = jnp.array(self.config.nn_eps_small, dtype=dtype)
        nn_eps_large = jnp.array(self.config.nn_eps_large, dtype=dtype)
        where_eps = jnp.array(self.config.where_eps, dtype=dtype)
        reps = jnp.array(1e-6, dtype=dtype)

        dtype_0 = jnp.array(0.0, dtype=dtype)
        dtype_005 = jnp.array(0.05, dtype=dtype)
        dtype_1 = jnp.array(1.0, dtype=dtype)
        dtype_2 = jnp.array(2.0, dtype=dtype)
        dtype_3 = jnp.array(3.0, dtype=dtype)
        dtype_4 = jnp.array(4.0, dtype=dtype)
        dtype_5 = jnp.array(5.0, dtype=dtype)
        dtype_half = jnp.array(0.5, dtype=dtype)

        cart = cart * jnp.reciprocal(self.config.sqrt_mass[:, None])
        diag = jnp.eye(npar_i, dtype=dtype)
        weight = dtype_1 - diag
        distvec = cart[None, :, :] - cart[:, None, :]
        distsq = jnp.sum(jnp.square(distvec), axis=2)
        distances = jnp.sqrt(distsq + diag)
        sph_coeff = jnp.abs(self.sph_coeff)[self.config.expand_indices]
        sph_coeff = (sph_coeff + reps) * where_eps
        sph = self.sph_cal(jnp.einsum("ijk, ij -> kij", distvec, jnp.reciprocal(distances + sph_coeff), optimize="optimal"))

        sph = sph * jnp.sqrt(dtype_4 * jnp.pi)

        emb_coeff = self.emb_coeff[self.config.expand_indices]
        exp_coeff = jnp.abs(self.exp_coeff)[self.config.expand_indices]
        sinc_x = jnp.pi * distances[:, :, None] * emb_coeff
        radial_func = jnp.cos(sinc_x) * jnp.exp(-exp_coeff * distances[:, :, None])
        scale_coeff = (self.scale_coeff * jnp.reciprocal(nradial_f))[self.config.expand_indices]
        radial_func = jnp.einsum("ijk, ijkn -> ijn", radial_func, scale_coeff, optimize="optimal")

        pindex_l = self.config.index_l[:pnorb_i]
        wradial = self.radialnn(radial_func).reshape(npar_i, npar_i, prmaxl_i+1, nwave_i)
        density = jnp.mean(wradial[:, :, -1] * weight[:, :, None], axis=1) 
        center_orbital = jnp.einsum("ijkn, kij, ij, inm -> ikm", \
                     wradial[:, :, pindex_l], \
                     sph[:pnorb_i], \
                     weight, \
                     (self.spec_coeff * jnp.reciprocal(jnp.sqrt(nwave_f) * (npar_f-dtype_1)))[self.config.spec_indices], \
                     optimize="optimal")

        weight_n = weight[nele_i:, :]
        ead_n = self.initnn_en(radial_func[nele_i:, :]) * weight_n[:, :, None]
        sph_n = sph[:pnorb_i, nele_i:, :]

        for iter_loop in range(MP_loop_i):

            norm_corb = center_orbital * self.config.ens_cg[:pnorb_i, None]
            density1 = jnp.sum(jnp.square(norm_corb), axis=1) * jnp.reciprocal(jnp.sqrt(prmaxl_f))
            density = jnp.concatenate((density, density1), axis=1)
            norm_corb = norm_corb * (self.config.ens_cg[:pnorb_i, None] * jnp.reciprocal(jnp.sqrt(dtype_2 * prmaxl_f)))
            radial = self.radialnn_en[iter_loop](ead_n).reshape(numatom_i, npar_i, 2, prmaxl_i, nwave_i)
            ead_n1 = jnp.einsum("ijkn, jkn, kij -> ijn", radial[:, :, 0, pindex_l], norm_corb, sph_n, optimize="optimal")
            ead_n1 = ead_n1 + jnp.einsum("ijkn, ikn, kij -> ijn", radial[:, :, 1, pindex_l], norm_corb[nele_i:], sph_n, optimize="optimal")
            ead_n = jnp.concatenate([ead_n, ead_n1 * weight_n[:, :, None]], axis=2)
         
            node_coeff = self.nodenn[iter_loop](density).reshape(npar_i, rmaxl_i, nwave_i) 
            radial = self.MPNN_list[iter_loop](radial_func).reshape(npar_i, npar_i, rmaxl_i+prmaxl_i, nwave_i)
            radial = radial * weight[:, :, None, None]
            
            contract_coeff_iter = (self.contract_coeff * jnp.reciprocal(jnp.sqrt(nwave_f)))[iter_loop, self.config.spec_indices]
            l_coeff_iter = self.l_coeff[iter_loop, self.config.spec_indices]
            center_orbital = self.sum_interaction(npar_i, nwave_i, prmaxl_i, pnorb_i, npar_f, nwave_f, rmaxl_f, pindex_l, dtype_2, reps, contract_coeff_iter, l_coeff_iter, radial, node_coeff, sph, center_orbital)

        norm_corb = center_orbital * self.config.ens_cg[:pnorb_i, None]
        density1 = jnp.sum(jnp.square(norm_corb), axis=1) * jnp.reciprocal(jnp.sqrt(prmaxl_f))
        density = jnp.concatenate((density, density1), axis=1)
        norm_corb = norm_corb * (self.config.ens_cg[:pnorb_i, None] * jnp.reciprocal(jnp.sqrt(dtype_2 * prmaxl_f)))
        radial = self.radialnn_en[-1](ead_n).reshape(numatom_i, npar_i, 2, prmaxl_i, nwave_i)
        ead_n1 = jnp.einsum("ijkn, jkn, kij -> ijn", radial[:, :, 0, pindex_l], norm_corb, sph_n, optimize="optimal")
        ead_n1 = ead_n1 + jnp.einsum("ijkn, ikn, kij -> ijn", radial[:, :, 1, pindex_l], norm_corb[nele_i:], sph_n, optimize="optimal")
        ead_n = jnp.concatenate([ead_n, ead_n1 * weight_n[:, :, None]], axis=2)

        orb_emb = self.orb_emb[:, :, self.config.spec_indices[nele_i:]]
        den_equi = center_orbital[nele_i:, :nrorb_i] 
        den_equi = jnp.broadcast_to(den_equi.reshape(1, 1, numatom_i, nrorb_i, nwave_i), (nele_i, ndet_i, numatom_i, nrorb_i, nwave_i))
        den_equi = jnp.concatenate([orb_emb, den_equi], axis=4)
        norbcoeff = []
        numorb = 0
        for l in range(reorbl_i):
            norbcoeff.append(self.norbcoeffnn[l](den_equi[:, :, :, numorb:numorb+2*l+1].reshape(nele_i, ndet_i, numatom_i, -1)))
            numorb = numorb + 2*l + 1
        for l in range(reorbl_i, rorbl_i):
            norbcoeff.append(self.norbcoeffnn[l](den_equi[:, :, :, numorb:numorb+2*l+1].reshape(nele_i, ndet_i, numatom_i, -1)) * self.hl_coeff[l] * hl_f)
            numorb = numorb + 2*l + 1
        norbcoeff = jnp.concatenate(norbcoeff, axis=3).reshape(nele_i, ndet_i, numatom_i, nrorb_i, nbasis_i)
        norbcoeff = norbcoeff.astype(det_dtype)

        ead_ne = ead_n[:, :nele_i]
        dist_ne = distances[nele_i:, :nele_i, None, None]
        radial = self.orbitalnn_en(ead_ne).reshape(numatom_i, nele_i, -1, nbasis_i)

        rindex_l = self.config.index_l[:nrorb_i]
        spin_func = self.emb_spin[self.config.spin_inp].reshape(nele_i, nele_i, ndet_i) + self.config.spin_inp.astype(dtype).reshape(nele_i, nele_i, 1)
 
        if self.config.eee:
            alpha_en = jnp.square(self.alpha_en(ead_ne).reshape(numatom_i, nele_i, rmaxl_i, nbasis_i) + jnp.sqrt(self.config.species[nele_i:, None, None] * jnp.reciprocal(jnp.arange(rmaxl_i, dtype=dtype)[:, None] + dtype_1)))
            radial_en = jnp.exp(-(alpha_en + en_eps) * dist_ne)
            contract_en = (self.contract_en * jnp.reciprocal(jnp.sqrt(nbasis_f)))[self.config.spec_indices[:nele_i]]
            enorb = jnp.einsum("kij, ijkn, ijkn, jknl -> kijl", \
                     sph[:, nele_i:, :nele_i], \
                     radial_en[:, :, self.config.index_l], \
                     radial[:, :, self.config.index_l], \
                     contract_en[:, self.config.index_l], \
                     optimize="optimal")

            radial_func_ee = radial_func[:nele_i, :nele_i]
            dist_ee = distances[:nele_i, :nele_i, None, None]
            ee_cusp = self.config.cusp[:nele_i, :nele_i, None, None, None]
            index_ee = self.config.expand_indices[:nele_i, :nele_i]
            fixed_ee = self.fixed_ee[index_ee] + ee_cusp
            e_lcoeff = self.e_lcoeff[:, self.config.spec_indices[:nele_i]]
            contract_sph = (self.contract_sph * jnp.reciprocal(jnp.sqrt(nwave_f) * (nele_f - dtype_1)))[self.config.spec_indices[:nele_i]]
            contract_ee = (self.contract_ee * jnp.reciprocal(jnp.sqrt(nwave_f)))[:, self.config.spec_indices[:nele_i]]

            iter_orb = center_orbital[:nele_i]
            save_orb = []
            density = density[:nele_i]
            for ieee in range(neee_i):
                radial_sph = self.orbitalnn_sph[ieee](radial_func_ee).reshape(nele_i, nele_i, rmaxl_i, nwave_i)
                radial_sph = radial_sph * weight[:nele_i, :nele_i, None, None]
                alpha_ee = self.alpha_ee[ieee](radial_func_ee).reshape(nele_i, nele_i, rmaxl_i, nwave_i)
                radial_ee = jnp.exp(-(jnp.square(alpha_ee) + fixed_ee[:, :, ieee]) * jnp.reciprocal(dtype_1 + dist_ee))
                node_coeff = self.nodenn_e[ieee](density).reshape(nele_i, rmaxl_i, nwave_i)
                init_orb = jnp.einsum(
                    "ijkn, ijkn, kij, jkn, iknp -> kip",
                    radial_sph[:, :, self.config.index_l],
                    radial_ee[:, :, self.config.index_l],
                    sph[:, :nele_i, :nele_i],
                    node_coeff[:, self.config.index_l],
                    contract_sph[:, self.config.index_l, ieee],
                    optimize="optimal")
                if self.config.ee_norm:
                    norm_factor = jnp.einsum("jik, jik -> i", init_orb, init_orb, optimize="optimal") * jnp.reciprocal(rmaxl_f*rmaxl_f*nwave_f)
                    init_orb = init_orb * jnp.reciprocal(jnp.sqrt(norm_factor + reps))[:, None]

                radial = self.orbitalnn_ee[ieee](radial_func_ee).reshape(nele_i, nele_i, prmaxl_i, nwave_i)
                radial = radial * weight[:nele_i, :nele_i, None, None]
                radial = radial[:, :, pindex_l]
                index_orbital1 = init_orb[self.config.index_i1]
                iter_orb = jnp.einsum("jkl, ijkl -> kil", \
                           iter_orb, \
                           radial, \
                           optimize="optimal") * jnp.reciprocal(jnp.sqrt(nele_f-dtype_1))
                index_orbital2 = iter_orb[self.config.index_i2]
                iter_orb = jnp.einsum("kij, kij, k, mk, imj, nm, injp -> inp", \
                        index_orbital1, \
                        index_orbital2, \
                        self.config.ens_cg, \
                        self.config.S_index_den, \
                        e_lcoeff[ieee][:, self.config.index_squ], \
                        self.config.S_index_add, \
                        contract_ee[ieee][:, pindex_l], \
                        optimize="optimal")
                save_orb.append(iter_orb)
                norm_ceorb = iter_orb * self.config.ens_cg[:pnorb_i, None]
                density1 = jnp.sum(jnp.square(norm_ceorb), axis=1) * jnp.reciprocal(jnp.sqrt(prmaxl_f))
                density = jnp.concatenate((density, density1), axis=1)

            radial_sph = self.orbitalnn_sph[-1](radial_func_ee).reshape(nele_i, nele_i, rmaxl_i, nwave_i)
            radial_sph = radial_sph * weight[:nele_i, :nele_i, None, None]
            alpha_ee = self.alpha_ee[-1](radial_func_ee).reshape(nele_i, nele_i, rmaxl_i, nwave_i)
            radial_ee = jnp.exp(-(jnp.square(alpha_ee) + fixed_ee[:, :, -1]) * jnp.reciprocal(dtype_1 + dist_ee))
            node_coeff = self.nodenn_e[-1](density).reshape(nele_i, rmaxl_i, nwave_i)
            init_orb = jnp.einsum(
                "ijkn, ijkn, kij, jkn, iknp -> ikp",
                radial_sph[:, :, self.config.index_l],
                radial_ee[:, :, self.config.index_l],
                sph[:, :nele_i, :nele_i],
                node_coeff[:, self.config.index_l],
                contract_sph[:, self.config.index_l, -1],
                optimize="optimal")
            if self.config.ee_norm:
                norm_factor = jnp.einsum("ijk, ijk -> i", init_orb, init_orb, optimize="optimal") * jnp.reciprocal(rmaxl_f*rmaxl_f*nwave_f)
                init_orb = init_orb * jnp.reciprocal(jnp.sqrt(norm_factor + reps))[:, None, None]

            iter_orb = jnp.concatenate([init_orb[:, :pnorb_i]] + save_orb, axis=2)
            eeorbcoeff = self.eeorbcoeffnn(iter_orb[:, :nrorb_i].reshape(nele_i, -1)).reshape(nele_i, ndet_i, nrorb_i, nele_i, nbasis_i).astype(det_dtype)
            contract_basis = (self.contract_basis * jnp.reciprocal(jnp.sqrt((self.config.neee+dtype_1)*nwave_f)))[self.config.spec_indices[:nele_i]]
            eeorb = jnp.einsum("ikn, iknm -> kim", iter_orb, contract_basis[:, pindex_l], optimize="optimal")

            index_i1 = self.config.index_i1[:nrealorb_i]
            index_i2 = self.config.index_i2[:nrealorb_i]
            ens_cg = self.config.ens_cg[:nrealorb_i].astype(det_dtype)
            S_index_den_p = self.config.S_index_den_p.astype(det_dtype)
            S_index_add_p = self.config.S_index_add_p.astype(det_dtype)
            l_orbcoeff = self.l_orbcoeffnn(ead_ne)[:, :, self.config.index_squ[:self.config.S_index_den_p.shape[0]]].astype(det_dtype)
            enorb = enorb.astype(det_dtype)
            eeorb = eeorb.astype(det_dtype)
            det = jnp.einsum("klij, kij, k, mk, lim, nm, edlnj, ednij -> die", \
                    enorb[index_i1], \
                    eeorb[index_i2], \
                    ens_cg, \
                    S_index_den_p, \
                    l_orbcoeff, \
                    S_index_add_p, \
                    norbcoeff, \
                    eeorbcoeff, \
                    optimize="optimal")
        else:
            alpha_en = jnp.square(self.alpha_en(ead_ne).reshape(numatom_i, nele_i, rorbl_i, nbasis_i) + jnp.sqrt(self.config.species[nele_i:, None, None] * jnp.reciprocal(jnp.arange(rorbl_i, dtype=dtype)[:, None] + dtype_1)))
            radial_en = jnp.exp(-(alpha_en + en_eps) * dist_ne)
            sph_ne = sph[:nrorb_i, nele_i:, :nele_i].astype(det_dtype)
            radial_en = radial_en[:, :, rindex_l].astype(det_dtype)
            radial = radial[:, :, rindex_l].astype(det_dtype)
            det = jnp.einsum("kij, ijkn, ijkn, edikn -> dje", \
                     sph_ne, \
                     radial_en, \
                     radial, \
                     norbcoeff, \
                     optimize="optimal")

        if self.config.spin_table==1:
            det = jnp.einsum("die, ied -> die", det, spin_func, optimize="optimal")

        det = (det * jnp.reciprocal(jnp.sqrt(nbasis_f * nrorb_f) * numatom_f))

         
        lognpsi = jnp.sum(jnp.zeros(1, dtype=dtype))
        if self.config.nn_table:
            weight_nn = weight[nele_i:, nele_i:]
            ead_nn = ead_n[:, nele_i:]

            nnout = self.outnn_alpha(ead_nn).reshape(numatom_i, numatom_i, 2, ngauss_i)

            distnn = distances[nele_i:, nele_i:, None]
            log_short = -jnp.sum((jnp.square(nnout[:, :, 0]) + nn_eps_small) * jnp.reciprocal(distnn) * weight_nn[:, :, None], axis=(0, 1))
            max_short = jnp.max(log_short)
            sub_short = log_short - max_short
            log_short = jnp.log(jnp.sum(jnp.exp(sub_short))) + max_short

            log_long = -(jnp.square(nnout[:, :, 1]) + nn_eps_large) * distnn
            # Copy a valid pair onto the diagonal for max, then exclude it from the sum.
            log_long = log_long * weight_nn[:, :, None] + log_long[0, 1] * (dtype_1 - weight_nn[:, :, None])
            max_long = jnp.max(log_long)
            sub_long = log_long - max_long
            log_long = jnp.log(jnp.sum(jnp.exp(sub_long) * weight_nn[:, :, None])) + max_long

            log_median = jnp.sum(self.outnn_coeff(ead_nn) * weight_nn[:, :, None]) * jnp.reciprocal(numatom_f * (numatom_f - dtype_1))

            lognpsi = log_short + log_median + log_long - jnp.log(ngauss_f * ngauss_f * numatom_f * (numatom_f - dtype_1))


        if nele_i < 1.5:
            psi = jnp.sum(det)
            sign = jnp.sign(psi)
            abs_psi = jnp.abs(psi)
            logepsi = jnp.log(abs_psi)
        else:
            if self.config.spin_table==2:
                if ndet_i < 1.5:
                    det = det.reshape(nele_i, nele_i)
                    sign_up, logup = jnp.linalg.slogdet(det[:nup_i, :nup_i])
                    sign_down, logdown = jnp.linalg.slogdet(det[nup_i:, nup_i:])
                    logepsi = logup + logdown
                else:
                    sign_up, up = jnp.linalg.slogdet(det[:, :nup_i, :nup_i])
                    sign_down, down = jnp.linalg.slogdet(det[:, nup_i:, nup_i:])
                    sign = sign_down * sign_up
                    sdet_ele = up + down
                    max_det = jnp.max(sdet_ele)
                    sub_det = sdet_ele-max_det
                    psi = jnp.einsum("i, i -> ", sign, jnp.exp(sub_det), optimize="optimal")
                    abs_psi = jnp.abs(psi)
                    logepsi = jnp.log(abs_psi) + max_det
            else:
                if ndet_i < 1.5:
                    det = det.reshape(nele_i, nele_i)
                    sign, logepsi = jnp.linalg.slogdet(det)
                else:
                    sign, sdet_ele = jnp.linalg.slogdet(det)
                    max_det = jnp.max(sdet_ele)
                    sub_det = sdet_ele-max_det
                    psi = jnp.einsum("i, i -> ", sign, jnp.exp(sub_det), optimize="optimal")
                    abs_psi = jnp.abs(psi)
                    logepsi = jnp.log(abs_psi) + max_det


        #distances = jnp.sqrt(distsq)
        #cusp = jnp.sum(-self.config.cusp.reshape(-1) / (dtype_1 + distances))
        logpsi = logepsi + lognpsi# + cusp
        return logpsi
 
    def sum_interaction(self, npar_i, nwave_i, prmaxl_i, pnorb_i, npar_f, nwave_f, rmaxl_f, pindex_l, dtype_2, reps, contract_coeff, l_coeff, radial, node_coeff, sph, center_orbital):

        iter_orb = jnp.einsum(
            "mjq, mjqk, imjk -> jik",
            center_orbital,
            contract_coeff[:, pindex_l],
            radial[:, :, pindex_l],
            optimize="optimal")
        iter_orb = iter_orb * jnp.reciprocal(npar_f)

        init_orb = jnp.einsum(
            "imjk, jim, mjk, ijkn -> jin",
            radial[:, :, prmaxl_i+self.config.index_l],
            sph,
            node_coeff[:, self.config.index_l],
            contract_coeff[:, prmaxl_i+prmaxl_i+self.config.index_l],
            optimize="optimal")

        if self.config.use_norm:
            norm_factor = jnp.einsum("jik, jik -> i", init_orb, init_orb, optimize="optimal") * jnp.reciprocal(rmaxl_f*rmaxl_f*nwave_f)  
            init_orb = init_orb * jnp.reciprocal(jnp.sqrt(norm_factor + reps))[:, None]
            #init_orb = init_orb * jnp.reciprocal(jnp.sqrt(jnp.mean(jnp.square(init_orb))))
        else:
            init_orb = init_orb * jnp.reciprocal(npar_f)

        index_orbital1 = init_orb[self.config.index_i1]
        index_orbital2 = iter_orb[self.config.index_i2]
        mp_orbital = jnp.einsum("kij, kij, k, mk, imj, nm -> inj", \
                     index_orbital1, \
                     index_orbital2, \
                     self.config.ens_cg, \
                     self.config.S_index_den, \
                     l_coeff[:, self.config.index_squ], \
                     self.config.S_index_add, \
                     optimize="optimal")
        
        center_orbital = jnp.einsum("ijk, ijkm -> ijm", center_orbital, contract_coeff[:, prmaxl_i+pindex_l], optimize="optimal")
        center_orbital = (center_orbital + mp_orbital) * jnp.reciprocal(jnp.sqrt(dtype_2))

        return center_orbital
        
