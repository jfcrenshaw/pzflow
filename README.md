![build](https://github.com/jfcrenshaw/python-package/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/jfcrenshaw/python-package/branch/main/graph/badge.svg?token=3P3LYKA7EV)](https://codecov.io/gh/jfcrenshaw/python-package)

# python-package
Template repo for a python package

Steps:
1. Change metadata and options in setup.cfg, including setting a new package name
2. Rename 'package' directory to the new package name
3. Change all occurences of 'package' in .github/workflows/main.yml to the new package name
4. Go to [codecov.io](https://codecov.io/), add this repo, then click setting->badge and copy the markdown code for the badge. Paste that above.
5. You can then install the package in edit mode (optionally with the `interactive flag`)  by cloning the repo, and running `pip install -e .[interactive]`
