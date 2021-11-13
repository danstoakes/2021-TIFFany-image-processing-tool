# TIFFany
TIFFany is a Python package which aims to perform various different computer vision tasks, with a primary focus on the identification and comparison of differences between images. Differences are calculated through implementations of common image processing tasks, such as mean-square error (MSE) and structural similarity index (SSIM). Visual operations such as Gaussian blur, median blur, absolute difference, etc are also available as operations.

# Installation
Once downloaded, the package can be installed globally by navigating to the parent folder and executing the command:

`$ pip install TIFFany`

# Requirements
Python 3 is the recommended version to be used with this tool, following the official deprecation of Python 2.

TIFFany has a heavy reliance on the `numPy` and `python-opencv` packages, which should be installed automatically via the `setup.py` script. If these packages do not install automatically, they can be installed with the command:

`$ pip install -r requirements.txt`

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
