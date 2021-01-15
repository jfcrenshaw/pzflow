from setuptools import setup, find_packages

extras = {"dev": ["pytest", "black", "pylint"], "docs": ["sphinx", "sphinx-rtd-theme"]}

setup(
    name="pzflow",
    version="1.0.0",
    author="John Franklin Crenshaw",
    author_email="jfc20@uw.edu",
    description="Modeling galaxy redshifts and magnitudes with normalizing flows",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="http://github.com/jfcrenshaw/pzflow",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["jax", "jaxlib", "pandas", "dill"],
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