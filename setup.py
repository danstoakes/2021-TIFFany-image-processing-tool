from setuptools import setup

setup(
    name="TIFFany",
    version="0.1",
    description="Image processing package which performs MSE, SSIM and various visual operations.",
    url="https://github.com/danstoakes/2021-TIFFany-image-processing-tool",
    author="Dan Stoakes",
    author_email="dan.stoakes8@gmail.com",
    license="MIT",
    packages=find_packages(include=["TIFFany"]),
    install_requires=[
        "opencv-python",
        "numpy"
    ],
    zip_safe=False
)
