import os
import pandas as pd
from pzflow import Flow


def two_moons_data() -> pd.DataFrame:
    """Return DataFrame with two moons example data.

    Two moons data originally from scikit-learn,
    i.e., `sklearn.datasets.make_moons`.
    """
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, "data/two-moons-data.pkl")
    data = pd.read_pickle(data_path)
    return data


def galaxy_data() -> pd.DataFrame:
    """Return DataFrame with examples galaxy data.

    100,000 galaxies from the buzzard simulation, with redshifts
    in the range (0,2.3) and photometry in the LSST ugrizy bands.
    """
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, "data/galaxy-data.pkl")
    data = pd.read_pickle(data_path)
    return data


def example_flow() -> Flow:
    """Return a normalizing flow that was trained on galaxy data.

    This flow was trained in the `redshift_example.ipynb` Jupyter notebook,
    on the example data available in `pzflow.examples.galaxy_data`.
    For more info: `print(example_flow().info)`.
    """
    this_dir, _ = os.path.split(__file__)
    flow_path = os.path.join(this_dir, "data/example-flow.dill")
    flow = Flow(file=flow_path)
    return flow
