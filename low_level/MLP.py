from flax import linen as nn
import jax.numpy as jnp
from typing import Optional, List
import jax

class ScaledDense(nn.Module):
    """
    A Dense layer with custom scaling. Its parameter shapes are defined
    explicitly in setup() using in_features to be robust.
    """
    in_features: int
    features: int
    cst: float = 1.0
    ls: float = 1.0
    use_bias: bool = False
    bias_init_value: Optional[jnp.ndarray] = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        """Define parameters with explicit shapes to avoid data dependencies."""
        self.scale = jnp.array(self.cst, dtype=self.dtype) / (jnp.sqrt(jnp.array(self.in_features*self.ls*self.ls, dtype=self.dtype))) 

        self.kernel = self.param(
            'kernel',
            nn.initializers.normal(1.0),
            (self.in_features, self.features),
            self.dtype
        )

        if self.use_bias:
            def bias_init_fn(rng):
                if self.bias_init_value is not None:
                    assert self.bias_init_value.shape == (self.features,), \
                        f"bias_init_value shape mismatch, expected {(self.features,)}, got {self.bias_init_value.shape}"
                    return self.bias_init_value.astype(self.dtype)
                return jnp.zeros(self.features, dtype=self.dtype)
            self.bias = self.param('bias', bias_init_fn)

    def __call__(self, x):
        """Apply the pre-defined layers to the input."""
        assert x.shape[-1] == self.in_features, f"Input shape {x.shape} does not match layer's in_features {self.in_features}"
        
        out = x @ (self.kernel * self.scale)
        if self.use_bias:
            out += self.bias 
        return out


class ResidualBlock(nn.Module):
    """A self-contained residual block module."""
    features: int
    layers_per_block: int
    cst: float
    ls: float
    scale: float
    use_bias: bool
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        """Define and apply layers within the compact __call__."""
        residual = x

        # This loop now works correctly under the parent's @nn.compact
        for i in range(self.layers_per_block):
            x = nn.silu(x)
            x = ScaledDense(
                in_features=x.shape[-1],
                features=self.features,
                cst=self.cst,
                ls=self.ls,
                use_bias=self.use_bias,
                name=f'layer_{i}',
                dtype=self.dtype
            )(x)


        dtype_scale = jnp.array(self.scale, dtype=self.dtype)
        dtype_norm = jnp.sqrt(jnp.array(1.0+dtype_scale*dtype_scale, dtype=self.dtype))
        x = (x + residual * dtype_scale) / dtype_norm
        return x


class MLP(nn.Module):
    """
    A Multi-Layer Perceptron using the @nn.compact pattern consistently.
    All layers are defined inline within the __call__ method.
    """
    num_output: int = 1
    num_blocks: int = 1
    features: int = 128
    layers_per_block: int = 2
    cst: float = 1.0
    ls: float = 1.0
    use_bias: bool = False
    use_linear: bool = False
    bias_init_value: Optional[jnp.ndarray] = None
    dtype: jnp.dtype = jnp.float32
    
    # This module now exclusively and correctly uses @nn.compact.
    # There is no setup() method for defining layers.
    @nn.compact
    def __call__(self, x):
        if not self.use_linear:
            # Define input layer here
            x = ScaledDense(
                in_features=x.shape[-1],
                features=self.features,
                cst=self.cst,
                ls=self.ls,
                use_bias=self.use_bias,
                name='input_layer',
                dtype=self.dtype
            )(x)

            # Use a standard Python for-loop to instantiate blocks.
            # @nn.compact ensures each block gets a unique name and parameters.
            for i in range(self.num_blocks):
                x = ResidualBlock(
                    features=self.features,
                    layers_per_block=self.layers_per_block,
                    cst=self.cst,
                    ls=self.ls,
                    scale=float(self.num_blocks),
                    use_bias=self.use_bias,
                    dtype=self.dtype,
                    name=f'block_{i}'
                )(x)

            x = nn.silu(x)
            x = ScaledDense(
                in_features=x.shape[-1],
                features=self.num_output,
                cst=1.0,
                ls=self.ls,
                use_bias=self.use_bias,
                bias_init_value=self.bias_init_value,
                name='output_layer',
                dtype=self.dtype
            )(x)
        else:
            # For the simple linear case, define the layer directly
            #x = ScaledDense(
            #    in_features=x.shape[-1],
            #    features=self.features,
            #    cst=1.0,
            #    use_bias=self.use_bias,
            #    bias_init_value=self.bias_init_value,
            #    name='input_layer',
            #    dtype=self.dtype
            #)(x)

            x = ScaledDense(
                in_features=x.shape[-1],
                features=self.num_output,
                cst=1.0,
                ls=self.ls,
                use_bias=self.use_bias,
                bias_init_value=self.bias_init_value,
                name='output_layer',
                dtype=self.dtype
            )(x)
        return x
