# Tutorials

Below are example notebooks demonstrating how to use PZFlow.
Each contains a link to open the notebook on Google Colab, as well as a link to the source code on Github.

### Basic

- [Introduction to PZFlow](intro.ipynb) - building a basic flow to model the two moons data set from `sklearn`
- [Chaining Bijectors](redshift_example.ipynb) - chaining multiple bijectors together, demonstrated on a photometric galaxy catalog
- [FlowEnsembles](ensemble_demo.ipynb) - using `FlowEnsemble` to create an ensemble of normalizing flows

### Intermediate

- [Convolving Gaussian Errors](gaussian_errors.ipynb) - convolving Gaussian errors during training and posterior calculation
- [Modeling Discrete Variables](dequantization.ipynb) - using dequantizers to model discrete data
- [Modeling Variables with Periodic Topology](spherical_flow_example.ipynb) - using circular splines to model data with periodic topology, e.g. positions on a sphere

### Advanced

- [Marginalizing Variables](marginalization.ipynb) - marginalizing over missing variables during posterior calculation
- [Convolving Non-Gaussian Errors](nongaussian_errors.ipynb) - convolving non-Gaussian errors during training and posterior calculation
