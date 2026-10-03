import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap


class Sampling():
    def __init__(self, sqrt_mass, model, MC_time=50, jnp_dtype="float32",
                 nwalker=4096, step=(0.1, 0.01, 1.0), prob=0.5, nmove=4):
        self.nele = int(jnp.sum(jnp.less(sqrt_mass, 2)))
        self.npar = sqrt_mass.shape[0]
        self.nuc = self.npar - self.nele
        if self.nuc != 4 or self.nele != 10:
            raise ValueError("Expected 10 electrons followed by N, H, H, H.")
        if MC_time < 1 or not 0 < prob < 1 or nmove < 1:
            raise ValueError("Check MC_time, prob and nmove.")
        if len(step) != 3 or not np.all(np.isfinite(step)) or np.any(np.asarray(step) < 0):
            raise ValueError("step = [electron step (bohr), r step (bohr), x step (degree)].")

        self.MC_time = MC_time
        self.sqrt_mass = sqrt_mass
        self.prob = prob
        self.nwalker = nwalker
        self.nmove = min(self.nele, nmove)
        self.wf_model = model
        self.vmap_model = vmap(model, in_axes=(None, 0))
        self.step = jnp.array(step, dtype=sqrt_mass.dtype) * 0.01
        self.MC = self.MC_sample()

    def nuclear_geometry(self, r, x):
        # x is the N-H angle to +z, not the H-N-H angle.
        chi = jnp.deg2rad(x)
        phi = jnp.deg2rad(jnp.array([0.0, 120.0, 240.0], dtype=r.dtype))
        nuclei = jnp.zeros((4, 3), dtype=r.dtype)
        nuclei = nuclei.at[1:, 0].set(r * jnp.sin(chi) * jnp.cos(phi))
        nuclei = nuclei.at[1:, 1].set(r * jnp.sin(chi) * jnp.sin(phi))
        height = jnp.where(x == 90.0, jnp.zeros_like(r), r * jnp.cos(chi))
        nuclei = nuclei.at[1:, 2].set(height)
        return nuclei

    def get_xr(self, coor):
        nh = coor[..., self.nele+1:, :] - coor[..., self.nele:self.nele+1, :]
        r = jnp.mean(jnp.linalg.norm(nh, axis=-1), axis=-1)
        rho = jnp.mean(jnp.linalg.norm(nh[..., :2], axis=-1), axis=-1)
        height = jnp.mean(nh[..., 2], axis=-1)
        x = jnp.rad2deg(jnp.arctan2(rho, height))
        return x, r

    def __call__(self, key, coor, init_logpsi, params, step, scale=1.0, adapt_rate=0.1):
        coor = jnp.asarray(coor, dtype=self.sqrt_mass.dtype)
        step = jnp.asarray(step, dtype=coor.dtype)
        adapt_rate = jnp.asarray(adapt_rate, dtype=coor.dtype)
        mwcoor = coor * self.sqrt_mass[:, None]
        init_logpsi = self.vmap_model(params, mwcoor)
        scale = jnp.asarray(scale, dtype=init_logpsi.dtype)
        x, r = self.get_xr(coor)
        coor, init_logpsi, key, step, x, r = self.MC(
            params, coor, init_logpsi, key, step, scale, x, r, adapt_rate)
        return key, coor, init_logpsi, step, x, r

    def MC_sample(self):
        def _MC_step(params, initcoor, init_logpsi, key, step, scale, x, r, mode):
            key = jax.random.split(key, 4)
            new_x, new_r = x, r
            if mode == 0:
                move_idx = jax.random.choice(key[1], self.nele, (self.nmove,), replace=False)
                noise = jax.random.normal(key[0], (self.nmove, 3), dtype=initcoor.dtype)
                coor = initcoor.at[move_idx].add(noise * step[0])
                valid = jnp.array(True)
            else:
                noise = jax.random.normal(key[0], (), dtype=initcoor.dtype)
                if mode == 1:
                    new_r = r + noise * step[1]
                else:
                    new_x = x + noise * step[2]
                valid = jnp.isfinite(new_r) & (new_r > 0.0) & (new_x > 0.0) & (new_x < 180.0)
                # Keep only the physical coordinate domain, with no sampling window.
                new_r = jnp.where(valid, new_r, r)
                new_x = jnp.where(valid, new_x, x)
                coor = initcoor.at[self.nele:].set(self.nuclear_geometry(new_r, new_x))

            mwcoor = coor * self.sqrt_mass[:, None]
            logpsi = self.wf_model(params, mwcoor)
            log_alpha = jnp.log(jax.random.uniform(key[2], (), dtype=init_logpsi.dtype))
            # Flat dr dx measure on this C3v family, with x in degrees.
            judge = ((2.0 * (logpsi-init_logpsi) / scale > log_alpha) |
                     ~jnp.isfinite(init_logpsi))
            judge = judge & valid & jnp.isfinite(logpsi)
            initcoor = jnp.where(judge, coor, initcoor)
            init_logpsi = jnp.where(judge, logpsi, init_logpsi)
            x = jnp.where(judge, new_x, x)
            r = jnp.where(judge, new_r, r)
            return initcoor, init_logpsi, judge, key[3], x, r

        vmap_mc = vmap(_MC_step, in_axes=(None, 0, 0, 0, None, None, 0, 0, None))

        def jax_scan(params, coor, logpsi, key, step, scale, x, r, adapt_rate):
            rates = []
            for mode in (0, 1, 2):
                def scan_fun(_, carry):
                    coor, logpsi, key, x, r, sum_move = carry
                    coor, logpsi, judge, key, x, r = vmap_mc(
                        params, coor, logpsi, key, step, scale, x, r, mode)
                    sum_move += jnp.sum(judge, dtype=coor.dtype)
                    return coor, logpsi, key, x, r, sum_move

                sum_move = jnp.array(0.0, dtype=coor.dtype)
                coor, logpsi, key, x, r, sum_move = jax.lax.fori_loop(
                    0, self.MC_time, scan_fun, (coor, logpsi, key, x, r, sum_move))
                rate = jax.lax.psum(sum_move, axis_name="GPUs") / self.nwalker / self.MC_time
                rates.append(rate)

            # Adapt all three widths, including during production sampling.
            rate = jnp.stack(rates)
            step = step * jnp.exp(adapt_rate * (rate-self.prob))
            step = jnp.maximum(step, self.step).astype(coor.dtype)
            return coor, logpsi, key, step, x, r

        return jax_scan
