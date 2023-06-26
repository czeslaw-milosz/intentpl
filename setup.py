from setuptools import find_packages
from setuptools import setup

setup(
    name="intention",
    version="0.0.0",
    description="Final project for NLP 2023 course @ MIM UW: "
                "Intent classification on MASSIVE dataset.",
    author="ASDF",
    author_email='asdf@asdf.com',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        'sacremoses'
    ],
    setup_requires=['wheel']
)
