.. MontePython documentation master file, created by
   sphinx-quickstart on Fri Mar 11 21:19:45 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MontePython's documentation!
=======================================

Methods for Object-based and Neighborhood Threat Evaluation in Python (MontePython) is a Python module for meteorological-based image segementation, matching, and tracking. This package was originally built for `NOAA's Warn-on-Forecast System <https://www.nssl.noaa.gov/projects/wof/>`_ (WoFS). 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Install
==================

MontePython is not available through PyPi at the moment. To install it, 
require cloning the environment and then when the parent directory, 
using the setup.py to install it. 

    git clone https://github.com/WarnOnForecast/MontePython

    python setup.py install


Content
===========

.. toctree::
   :maxdepth: 2 
   
   Object Identification <object_id>
   Object Quality Control <object_qc>
   Object Matching <object_matching>
   Object Tracking <object_tracking>
   Storm Mode Classification <storm_mode>
   

