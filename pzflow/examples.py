import os

import pandas as pd

from pzflow import Flow


def two_moons_data() -> pd.DataFrame:
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, "data/two-moons-data.pkl")
    data = pd.read_pickle(data_path)
    return data


def galaxy_data() -> pd.DataFrame:
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, "data/galaxy-data.pkl")
    data = pd.read_pickle(data_path)
    return data


def example_flow() -> Flow:
    this_dir, _ = os.path.split(__file__)
    flow_path = os.path.join(this_dir, "data/example-flow.dill")
    flow = Flow(file=flow_path)
    return flow
