# SchrödingerNet
## Introduction:
This repository provides an implementation of SchrödingerNet using equivariant Message Passing Neural Networks (MPNNs) to represent the wavefunction, built on the JAX framework. SchrödingerNet offers a novel approach to solving the full electronic-nuclear Schrödinger equation (SE) by defining a custom loss function designed to equalize local energies throughout the system. The method adopts a symmetry-adapted wavefunction ansatz that preserves translational, rotational, and permutational symmetries while incorporating both nuclear and electronic coordinates. This framework allows for the generation of accurate potential energy surfaces and incorporates non-BOA corrections.
## Requirements:
* jax+flax
* optax

## Examples:
Training can be done by executing the command "python3 $path. 'pretrain'" where path is the path where the code is saved. The command will call the "train.py" script in the "pretrain" and "run" folder. These parameters for training are set in “input.inp” and the initial dataset is saved in a file named “configuration”. Examples of input.in and configurations can be found in the examples folder. The code for evaluating electron energies and diagonal Born-Oppenheimer corrections (DBOC) can be found in the eval folder.
