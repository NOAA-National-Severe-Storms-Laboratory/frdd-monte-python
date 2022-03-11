![Unit Tests](https://github.com/WarnOnForecast/MontePython/actions/workflows/continuous_intergration.yml/badge.svg)

# MontePython


Methods for Object-based and Neighborhood Threat Evaluation in Python (MontePython) is a Python module for meteorological-based image segementation, matching, and tracking. This package was originally built for [NOAA's Warn-on-Forecast System](https://www.nssl.noaa.gov/projects/wof/)(WoFS). 



### Requirements
- numpy 
- scipy
- scikit-image
- scikit-learn 
- pandas 

## Object Identification 

### Single Threshold 
```python 
import monte_python 

# Create some fake storms
centers = [(40, 45), (40, 58), (65, 90), (90, 55), (40,20)]
storms,x,y = monte_python.create_fake_storms(center) 

storm_labels, object_props = monte_python.label( input_data = storms,
                                   method ='single_threshold', 
                                   return_object_properties=True, 
                                   params = {'bdry_thresh':40} )                                    
```

### Watershed 
```python
storm_labels, object_props = monte_python.label(  input_data = storms, 
                       method ='watershed', 
                       return_object_properties=True, 
                       params = {'min_thresh':25,
                                 'max_thresh':80,
                                 'data_increment':20,
                                 'area_threshold': 150,
                                 'dist_btw_objects': 50} 
                       )
```

### Iterative watershed
```python
# For this example, we are using two iterations of the iterative watershed method. 
# However, if the expected objects existed across more spatial scales, we could introduce more
# iterations. 
param_set = [ {'min_thresh':10,
                                 'max_thresh':80,
                                 'data_increment':20,
                                 'area_threshold': 200,
                                 'dist_btw_objects': 50} , 
            
              {'min_thresh':25,
                                 'max_thresh':80,
                                 'data_increment':20,
                                 'area_threshold': 50,
                                 'dist_btw_objects': 10} 
            ]

params = {'params': param_set }

# This is not an important set for the watershed, but 
# simply something to make this fake example work. 
input_data = np.where(storms > 10, storms, 0)
storm_labels, object_props = monte_python.label(  input_data = input_data, 
                       method ='iterative_watershed', 
                       return_object_properties=True, 
                       params = params,  
                       )
```
