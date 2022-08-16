![Unit Tests](https://github.com/WarnOnForecast/MontePython/actions/workflows/continuous_integration.yml/badge.svg)

# MontePython


Methods for Object-based and Neighborhood Threat Evaluation in Python (MontePython) is a Python module for meteorological-based image segementation, matching, quality control and tracking. This package was originally built for [NOAA's Warn-on-Forecast System](https://www.nssl.noaa.gov/projects/wof/) (WoFS), but is applicable to any gridded data. MontePython also includes a 7-scheme storm identification and classification scheme from [Potvin et al. (2022)](https://journals.ametsoc.org/view/journals/atot/39/7/JTECH-D-21-0141.1.xml?rskey=dHTUIB&result=5).


## Object Identification 

There are three different object identification (ID) methods in MontePython: single threhsold, enhanced watershed, and iterative watershed. To see examples of how to use these different methods, see the [object ID tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/object_id.ipynb)

## Object Quality Control 

Object identification methods are imperfect and as such a post-processing step is often required to improve the identification. MontePython's ObjectQualityControler has multiple options for qc'ing the output of object identification. To see examples of differnt quality control methods, see the [object quality control tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/object_quality_control.ipynb)


## Object Matching 

MontePython can perform matching between two sets of objects (e.g., forecasts and observations). The default matching is based on a total interest score in [Skinner et al. (2018)](https://journals.ametsoc.org/view/journals/wefo/33/5/waf-d-18-0020_1.xml), which includes minimum displacement, centroid displacement, and time displacement. The user can also provide a custom scoring function. Additionally, the code can also handle non-one-to-one matching. This is applicable to situations where we might want to associate multiple observed objects with a single forecast object. To see examples of how to perform object matching, see the [object matching tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/object_matching.ipynb)

## Object Tracking

Once identified, objects over time can be tracked using MontePython's ObjectTracker. The algorithm is designed for high temporal resolution data such that matches in time can be largely determined by the amount of overlap. The tracking algorithm can handle both splits and mergers. To see examples of how to perform object tracking, see the [object tracking tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/object_tracking.ipynb)

## Storm Identification and Classification 
[Potvin et al. (2022)](https://journals.ametsoc.org/view/journals/atot/39/7/JTECH-D-21-0141.1.xml?rskey=dHTUIB&result=5) developed a storm identification and classification method, which is applicable to both convection-allowing model output and observed radar. To see examples of how to use this method , see the [storm mode tutorial notebook](https://github.com/WarnOnForecast/MontePython/blob/master/tutorial_notebooks/storm_mode_classification.ipynb)

### Requirements
- numpy 
- scipy
- scikit-image
- scikit-learn 
- pandas 
- numba 

