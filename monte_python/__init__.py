# Handle importing to make package more user-friendly 
from .object_identification import label, quantize_probabilities 
from .object_quality_control import QualityControler 
from .object_matching import ObjectMatcher
from .object_tracking import ObjectTracker 
from .storm_mode_classification import StormModeClassifier

import os
__key__ = 'PACKAGE_VERSION'
__version__= os.environ[__key__] if __key__ in os.environ else '1.1.0'
