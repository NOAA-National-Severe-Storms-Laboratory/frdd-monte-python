# Always prefer setuptools over distutils

import setuptools  # this is the "magic" import

import monte_python

from numpy.distutils.core import setup, Extension

#from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='monte_python', 
    version=monte_python.__version__,
    description='Methods for Object-based and Neighborhood Threat Evaluation in Python', 
    long_description=long_description,
    long_description_content_type='text/markdown',  
    url='https://github.com/WarnOnForecast/MontePython', 
    author='NOAA National Severe Storms Laboratory', 
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Scientists',
        'Programming Language :: Python :: 3'
    ],
    install_requires = [
        'scikit-learn',
        'scikit-image>=0.18.1',
        'matplotlib<=3.4.3', 
    ],
    packages=['monte_python', 'monte_python._plot'],  # Required
    python_requires='>=3.8, <4',
    package_dir={'monte_python': 'monte_python'},
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/WarnOnForecast/MontePython/issues',
        'Source': 'https://github.com/WarnOnForecast/MontePython',
    },
)
