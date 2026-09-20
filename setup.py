import os
from setuptools import setup, find_packages

VERSION = '1.0.2'


setup(
    name = 'stamps',
    packages = find_packages(exclude=['contrib', 'docs', 'tests']),
    license = 'GPLv3',
    version = VERSION,
    description = 'Spatial Temporal Analysis and Mapping Python Suite',
    install_requires = [
        'numpy>=1.21', 'scipy>=1.7', 'pandas>=1.3', 'six>=1.15'],
    author = 'stemlab',
    author_email = 'stemlab689@gmail.com',
    url = 'https://github.com/stemlab689/stamps',
    download_url =\
        'https://gitlab.com/STEMLabTW/stamps/-/archive/{v}/stamps-{v}.zip'\
        .format(v=VERSION),
    keywords = 'analysis mapping stamps',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ]
    )
