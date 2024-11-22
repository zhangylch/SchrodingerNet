import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn

initializer = nn.initializers.normal()

class ResidualBlock(nn.Module):
    features: int
    nlayer: int
    layer_norm: bool 

    def setup(self):
        self.layers = [nn.Dense(self.features) for _ in range(self.nlayer)]

    def __call__(self, x):
        residual = x

        for layer in self.layers:
            x = nn.silu(x)
            x = layer(x)
           
        x += residual
        return x / jnp.sqrt(2.0)

class MLP(nn.Module):
    num_output: int = 1
    num_blocks: int = 1
    features: int = 128
    layers_per_block: int = 2
    layer_norm: bool = True
    bias_init_value: jnp.ndarray = None

    def setup(self):
        self.input_layer = nn.Dense(self.features)
        self.blocks = [ResidualBlock(self.features, self.layers_per_block, self.layer_norm) for _ in range(self.num_blocks)]

        bias_init_value = self.bias_init_value if self.bias_init_value is not None else jnp.zeros(self.num_output)
        
        # Initialize the bias with the given custom value
        self.bias = self.param('bias', lambda rng: bias_init_value)
        
        # Initialize the bias with the given custom value
        self.weights = self.param('weight', lambda rng, shape: initializer(rng, shape), (self.features, self.num_output))
        if self.layer_norm: self.LN = nn.LayerNorm()
        
    def __call__(self, x):
        if self.layer_norm: x = self.LN(x)
        x = self.input_layer(x)

        for block in self.blocks:
            x = block(x)
          
        x = nn.silu(x)
        return jnp.dot(x, self.weights) + self.bias

## Initialize and test the ResNet
#key = random.PRNGKey(0)
#input_shape = (1, 784)  # Batch size 1, 784 features (e.g., flattened 28x28 MNIST image)
#x = jnp.ones(input_shape)
#
#model = ResNet(num_output=10, num_blocks=3, features=128, layers_per_block=2, use_layer_norm=True)
#params = model.init(key, x)
#output = model.apply(params, x)
#print(output)

