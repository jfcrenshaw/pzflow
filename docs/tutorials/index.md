# Tutorials

Below are example notebooks demonstrating how to use PZFlow.
Each contains a link to open the notebook on Google Colab, as well as a link to the source code on Github.

### Basic

- [Introduction to PZFlow](intro.ipynb) - using the default flow to train, sample, and calculate posteriors
- [Conditional Flows](conditional_demo.ipynb) - building a conditional flow to model conditional distributions
- [Convolving Gaussian Errors](gaussian_errors.ipynb) - convolving Gaussian errors during training and posterior calculation
- [Flow Ensembles](ensemble_demo.ipynb) - using `FlowEnsemble` to create an ensemble of normalizing flows

### Intermediate

- [Customizing the flow](customizing_example.ipynb) - Customizing the bijector and latent space
- [Modeling Discrete Variables](dequantization.ipynb) - using dequantizers to model discrete data
- [Modeling Variables with Periodic Topology](spherical_flow_example.ipynb) - using circular splines to model data with periodic topology, e.g. positions on a sphere
- [Other Latent Distributions](tutorials/index.md) - using a user-specified latent distribution

### Advanced

- [Marginalizing Variables](marginalization.ipynb) - marginalizing over missing variables during posterior calculation
- [Convolving Non-Gaussian Errors](nongaussian_errors.ipynb) - convolving non-Gaussian errors during training and posterior calculation
