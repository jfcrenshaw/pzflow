import os

import pandas as pd

from pzflow import Flow


def get_twomoons_data() -> pd.DataFrame:
    """Return DataFrame with two moons example data.

    Two moons data originally from scikit-learn,
    i.e., `sklearn.datasets.make_moons`.
    """
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, "data/two-moons-data.pkl")
    data = pd.read_pickle(data_path)
    return data


def get_galaxy_data() -> pd.DataFrame:
    """Return DataFrame with example galaxy data.

    100,000 galaxies from the Buzzard simulation [1], with redshifts
    in the range (0,2.3) and photometry in the LSST ugrizy bands.

    References
    ----------
    [1] Joseph DeRose et al. The Buzzard Flock: Dark Energy Survey
    Synthetic Sky Catalogs. arXiv:1901.02401, 2019.
    https://arxiv.org/abs/1901.02401
    """
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, "data/galaxy-data.pkl")
    data = pd.read_pickle(data_path)
    return data


def get_city_data() -> pd.DataFrame:
    """Return DataFrame with example city data.

    The countries, names, population, and coordinates of 47,966 cities.

    Subset of the Kaggle world cities database.
    https://www.kaggle.com/max-mind/world-cities-database
    This database was downloaded from MaxMind. The license follows:

        OPEN DATA LICENSE for MaxMind WorldCities and Postal Code Databases

        Copyright (c) 2008 MaxMind Inc.  All Rights Reserved.

        The database uses toponymic information, based on the Geographic Names
        Data Base, containing official standard names approved by the United States
        Board on Geographic Names and maintained by the National
        Geospatial-Intelligence Agency. More information is available at the Maps
        and Geodata link at www.nga.mil. The National Geospatial-Intelligence Agency
        name, initials, and seal are protected by 10 United States Code Section 445.

        It also uses free population data from Stefan Helders www.world-gazetteer.com.
        Visit his website to download the free population data.  Our database
        combines Stefan's population data with the list of all cities in the world.

        All advertising materials and documentation mentioning features or use of
        this database must display the following acknowledgment:
        "This product includes data created by MaxMind, available from
        http://www.maxmind.com/"

        Redistribution and use with or without modification, are permitted provided
        that the following conditions are met:
        1. Redistributions must retain the above copyright notice, this list of
        conditions and the following disclaimer in the documentation and/or other
        materials provided with the distribution.
        2. All advertising materials and documentation mentioning features or use of
        this database must display the following acknowledgement:
        "This product includes data created by MaxMind, available from
        http://www.maxmind.com/"
        3. "MaxMind" may not be used to endorse or promote products derived from this
        database without specific prior written permission.

        THIS DATABASE IS PROVIDED BY MAXMIND.COM ``AS IS'' AND ANY
        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
        WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL MAXMIND.COM BE LIABLE FOR ANY
        DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
        (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
        LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
        ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
        DATABASE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, "data/city-data.pkl")
    data = pd.read_pickle(data_path)
    return data


def get_example_flow() -> Flow:
    """Return a normalizing flow that was trained on galaxy data.

    This flow was trained in the `redshift_example.ipynb` Jupyter notebook,
    on the example data available in `pzflow.examples.galaxy_data`.
    For more info: `print(example_flow().info)`.
    """
    this_dir, _ = os.path.split(__file__)
    flow_path = os.path.join(this_dir, "data/example-flow.pkl")
    flow = Flow(file=flow_path)
    return flow
