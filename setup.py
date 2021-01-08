import os
from setuptools import setup, find_packages

setup(
    name='chronos',
    version='0.1',
    author="Joshua Dempster",
    description="Time series modeling of CRISPR KO readcounts",
    packages=find_packages(),
    package_data={'': ['*.r']},
    # Install requirements via pip install -r requirements.txt
    # install_requires=[...]
)
