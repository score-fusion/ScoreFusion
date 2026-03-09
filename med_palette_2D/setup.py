from setuptools import setup, find_packages

setup(
    name='palette',
    version='0.0.1',
    description='install package for med palette',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)


