# setup.py : This file is used to set up the package and its dependencies. 
# It tells Python how to install the package and what dependencies it has. 
# This is important for making sure that anyone who wants to use the package can easily install it and get it working without having to manually install each dependency.

# WITHOUT this file:
#   from src.logger import logger    <- only works if you run from project root
#
# WITH this file (after pip install -e .):
#   from src.logger import logger    <- works from ANYWHERE
#   This is how production ML projects work
#
# -e means "editable mode" — code changes reflect immediately, no reinstall needed


from setuptools import setup, find_packages

setup(
    name='ragbot',
    version='0.1.0',
    description='A Retrieval-Augmented Generation (RAG) bot for document question answering.',
    author='Niral Patel',
    author_email='nir64.au@gmail.com',
    packages=find_packages(),
    install_requires=[
    ],
)