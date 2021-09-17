from setuptools import setup, find_packages

extras = {
    "dev": ["black", "pytest", "pylint", "twine"],
    "docs": ["sphinx", "sphinx-rtd-theme"],
}

# load the version number
with open("pzflow/version.py") as f:
    __version__ = f.read().replace('"', "").split("=")[1]

setup(
    name="pzflow",
    version=__version__,
    author="John Franklin Crenshaw",
    author_email="jfc20@uw.edu",
    description="Probabilistic modeling of tabular data with normalizing flows.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="http://github.com/jfcrenshaw/pzflow",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["dill", "jax", "jaxlib", "pandas>=1.1"],
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6.0",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)