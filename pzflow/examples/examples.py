import jax.numpy as np
from pzflow import Flow
import os


def example_data():
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, "galaxy-data.npy")
    columns = ("redshift", "u", "g", "r", "i", "z", "y")
    data = np.load(data_path)
    return columns, data


def example_flow():
    this_dir, _ = os.path.split(__file__)
    flow_path = os.path.join(this_dir, "example-flow.dill")
    flow = Flow(file=flow_path)
    flow.info = """
                This is an example flow, trained on 100,000 simulated galaxies with 
                redshifts in the range (0,2.3) and photometry in the LSST ugrizy bands.

                The flow expects (and produces) arrays with columns in the order
                (redshift, u, g, r, i, z, y)
                """.replace(
        "\n" + " " * 16, "\n"
    )
    return flow