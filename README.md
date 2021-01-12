![build](https://github.com/jfcrenshaw/pzflow/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/jfcrenshaw/pzflow/branch/main/graph/badge.svg?token=qR5cey0swQ)](https://codecov.io/gh/jfcrenshaw/pzflow)

# pzflow
Modeling galaxy redshifts and magnitudes with a normalizing flow

## Development

To work on pzflow, after cloning this repo:
1. Create a virtual environment with Python  
E.g., with conda `conda create -n pzflow`
2. Activate the environment.  
E.g., `conda activate pzflow`  
3. Install pzflow in edit mode with the `dev` flag  
I.e., in the root directory, `pip install -e .[dev]`

If you want to run pzflow on a GPU with CUDA, then you need to follow the GPU-enabled installation instructions for jaxlib [here](https://github.com/google/jax).  
You make also need to add the following to your `.bashrc`:
```
# cuda setup
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda/bin
```
