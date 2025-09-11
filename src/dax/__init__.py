"""PAX -- Particle filters in Jax.

Author:
    Ilias Bilionis

Date:   
    11/26/2024
"""


from .probability import *
from .markov import *
from .likelihood import *
from .particle_approximation import *
from .ssm import *
from .sde import *
from .filter import *
from .smooth import *
from .em import *
from .mcmc import *


from pathlib import Path
import toml

__version__ = "unknown"
pyproject_toml_file = Path(__file__).parent.parent / "pyproject.toml"
if pyproject_toml_file.exists() and pyproject_toml_file.is_file():
    data = toml.load(pyproject_toml_file)
    # check project.version
    if "project" in data and "version" in data["project"]:
        __version__ = data["project"]["version"]
    # check tool.poetry.version
    elif "tool" in data and "poetry" in data["tool"] and "version" in data["tool"]["poetry"]:
        __version__ = data["tool"]["poetry"]["version"]

# Once installed use the following
# import importlib.metadata
# __version__ = importlib.metadata.version("dax")