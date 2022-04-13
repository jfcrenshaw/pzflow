# Install

You can install PZFlow from PyPI with pip:

```shell
pip install pzflow
```

If you want to run PZFlow on a GPU with CUDA, you need to follow the GPU-enabled installation instructions for Jax [here](https://github.com/google/jax).
You may also need to add cuda to your path.
For example, I needed to add the following to my `.bashrc`:

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
