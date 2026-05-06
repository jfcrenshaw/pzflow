"""Import modules and set version."""

from importlib.metadata import version

from pzflow.flow import Flow as Flow
from pzflow.flowEnsemble import FlowEnsemble as FlowEnsemble

__version__ = version("pzflow")
