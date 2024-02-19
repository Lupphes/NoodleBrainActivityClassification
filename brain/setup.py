from setuptools import setup, find_packages

setup(
    name="brain_solver",
    version="0.9.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "albumentations",
        "matplotlib",
        "PyWavelets",
        "librosa",
        "scipy",
        "pytorch_lightning",
        "torchvision",
        "d2l",
    ],
)
