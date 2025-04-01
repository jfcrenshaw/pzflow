![build](https://github.com/jfcrenshaw/pzflow/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/jfcrenshaw/pzflow/branch/main/graph/badge.svg?token=qR5cey0swQ)](https://codecov.io/gh/jfcrenshaw/pzflow)
[![PyPI version](https://badge.fury.io/py/pzflow.svg)](https://badge.fury.io/py/pzflow)
[![DOI](https://zenodo.org/badge/327498448.svg)](https://zenodo.org/badge/latestdoi/327498448)
[![Docs](https://img.shields.io/badge/Docs-https%3A%2F%2Fjfcrenshaw.github.io%2Fpzflow%2F-red)](https://jfcrenshaw.github.io/pzflow/)

# PZFlow

PZFlow is a python package for probabilistic modeling of tabular data with normalizing flows.

If your data consists of continuous variables that can be put into a Pandas DataFrame, pzflow can model the joint probability distribution of your data set.

The `Flow` class makes building and training a normalizing flow simple.
It also allows you to easily sample from the normalizing flow (e.g., for forward modeling or data augmentation), and calculate posteriors over any of your variables.

There are several tutorial notebooks in the [docs](https://jfcrenshaw.github.io/pzflow/tutorials/).

## Installation

See the instructions in the [docs](https://jfcrenshaw.github.io/pzflow/install/).

## Citation

If you use this package in your research, please cite the following two sources:

1. The paper
```bibtex
@ARTICLE{2024AJ....168...80C,
       author = {{Crenshaw}, John Franklin and {Kalmbach}, J. Bryce and {Gagliano}, Alexander and {Yan}, Ziang and {Connolly}, Andrew J. and {Malz}, Alex I. and {Schmidt}, Samuel J. and {The LSST Dark Energy Science Collaboration}},
        title = "{Probabilistic Forward Modeling of Galaxy Catalogs with Normalizing Flows}",
      journal = {\aj},
     keywords = {Neural networks, Galaxy photometry, Surveys, Computational methods, 1933, 611, 1671, 1965, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2024,
        month = aug,
       volume = {168},
       number = {2},
          eid = {80},
        pages = {80},
          doi = {10.3847/1538-3881/ad54bf},
archivePrefix = {arXiv},
       eprint = {2405.04740},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024AJ....168...80C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

2. The [Zenodo deposit](https://zenodo.org/records/10710271) associated with the version number you used.
The Zenodo deposit


### Sources

PZFlow was originally designed for forward modeling of photometric redshifts as a part of the Creation Module of the [DESC](https://lsstdesc.org/) [RAIL](https://github.com/LSSTDESC/RAIL) project.
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
