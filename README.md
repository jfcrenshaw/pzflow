![build](https://github.com/jfcrenshaw/pzflow/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/jfcrenshaw/pzflow/branch/main/graph/badge.svg?token=qR5cey0swQ)](https://codecov.io/gh/jfcrenshaw/pzflow)
[![PyPI version](https://badge.fury.io/py/pzflow.svg)](https://badge.fury.io/py/pzflow)
[![DOI](https://zenodo.org/badge/327498448.svg)](https://zenodo.org/badge/latestdoi/327498448)

# pzflow

Probabilistic modeling of tabular data with normalizing flows.

If your data consists of continuous variables that can be put into a Pandas DataFrame, pzflow can model the joint probability distribution of your data set.

The `Flow` class makes building and training a normalizing flow simple.
It also allows you to easily sample from the normalizing flow (e.g., for forward modeling or data augmentation), and calculate posteriors over any of your variables.

There are many example notebooks demonstrating how to use pzflow.

Basic examples:
1. [Introduction of a basic flow with the two moons data set](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/intro.ipynb)
2. [A more complex example with galaxy redshifts](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/redshift_example.ipynb)
3. [Building a conditional flow on redshift data](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/conditional_redshift_example.ipynb)
4. [Convolving Gaussian errors during training and posterior calculation](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/gaussian_errors.ipynb)
5. [Using a uniform dequantizer to model discrete data](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/dequantization.ipynb)
6. [Using `FlowEnsemble` to build a deep ensemble](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/ensemble_demo.ipynb)

Advanced examples:
1. [Marginalizing over missing columns during posterior calculation](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/marginalization.ipynb)
2. [Convolving non-Gaussian erorrs during training and posterior calculation](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/nongaussian_errors.ipynb)
3. [Building a more complicated latent distribution to model data with a compact domain and periodic topology](https://colab.research.google.com/github/jfcrenshaw/pzflow/blob/main/examples/spherical_flow_example.ipynb)

If you notice any bugs, have any questions, or would like to request a feature, please [submit an issue](https://github.com/jfcrenshaw/pzflow/issues).

It is important to note that there are two different conventions in the literature for the direction of the bijection in normalizing flows.
pzflow defines the bijection as the mapping from the data space to the latent space, and the inverse bijection as the mapping from the latent space to the data space.
This distinction can be important when designing more complicated bijections (e.g., in Example 2 above).

## Citation

We are preparing a paper on pzflow.
If you use this package in your research, please check back here for a citation before publication.
In the meantime, please cite the [Zenodo release](https://zenodo.org/badge/latestdoi/327498448).

## Installation

You can install pzflow from PyPI with pip:
```shell
pip install pzflow
```
If you want to run pzflow on a GPU with CUDA, you need to follow the GPU-enabled installation instructions for jaxlib [here](https://github.com/google/jax).
You may also need to add the following to your `.bashrc`:
```shell
# cuda setup
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda/bin
```
If you have the GPU enabled version of jax installed, but would like to run on a CPU, add the following to the top of your scripts/notebooks:
```python
import jax
# Global flag to set a specific platform, must be used at startup.
jax.config.update('jax_platform_name', 'cpu')
```
Note that if you run jax on GPU in multiple Jupyter notebooks simultaneously, you may get `RuntimeError: cuSolver internal error`. Read more [here](https://github.com/google/jax/issues/4497) and [here](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html).

### Development

To work on pzflow, after forking and cloning the repo:
1. Create a virtual environment with Python  
E.g., with conda `conda create -n pzflow`
2. Activate the environment.  
E.g., `conda activate pzflow`  
3. Install pzflow in edit mode with the `dev` flag  
I.e., in the root directory, `pip install -e .[dev]`


### Sources

pzflow was originally designed for forward modeling of photometric redshifts as a part of the Creation Module of the [DESC](https://lsstdesc.org/) [RAIL](https://github.com/LSSTDESC/RAIL) project.
The idea to use normalizing flows for photometric redshifts originated with [Bryce Kalmbach](https://github.com/jbkalmbach).
The earliest version of the normalizing flow in RAIL was based on a notebook by [Francois Lanusse](https://github.com/eiffl) and included contributions from [Alex Malz](https://github.com/aimalz).

The functional jax structure of the bijectors was originally based on [`jax-flows`](https://github.com/ChrisWaites/jax-flows) by [Chris Waites](https://github.com/ChrisWaites). The implementation of the Neural Spline Coupling is largely based on the [Tensorflow implementation](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/rational_quadratic_spline.py), with some inspiration from [`nflows`](https://github.com/bayesiains/nflows/).

Neural Spline Flows are based on the following papers:

  > [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)\
  > Laurent Dinh, David Krueger, Yoshua Bengio\
  > _arXiv:1410.8516_

  > [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)\
  > Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio\
  > _arXiv:1605.08803_

  > [Neural Spline Flows](https://arxiv.org/abs/1906.04032)\
  > Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios\
  > _arXiv:1906.04032_
