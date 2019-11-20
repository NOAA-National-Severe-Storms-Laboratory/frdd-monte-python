###################################################
# DESCRIPTION OF THE WOFS PACKAGES
####################################################

1) LOADING DATA 
    Various data can be load from the /data in the /wofs directory. 
    -> loadEnsembleData.py loads the ensemble data from the summary files.
    -> loadMRMSData.py loads the data from the MRMS files. Directories for the MRMS files
                        can be found in /home/monte.flora/wofs/util/config.py
    -> loadWRFGrid.py loads data from the WRF output files. 
    -> loadLSRs.py loads the local storm reports 

2) EVALUATION
    Verification metrics are calculated in verification_metrics.py. This includes
    contingency tables metrics and reliability. 

3) PLOTTING
    Various plots can be drawn up from the Plot.py. This includes traditional
    spatial plots as well as performance, ROC, and reliability diagrams. 

4) PROCESSING
    /processing includes ObjectIdentification, ObjectMatching, and EnhancedWatershedSegmenter
    -> ObjectIdentification handles object identification through either single threshold or 
                            enhanced watershed algorithm (Lakshamanan et al. 2009). ObjectIdentification
                            also includes a class for quality controlling the objects identified. 
    -> ObjectMatching handles matching two sets of objects together using the total interest score (Davis et al. 2006) 
                        
5) UTILITIES
    /util includes basic, useful functions including the Patrick Skinner's newse post-processing scripts, 
          config script, multiprocessing scripts. 

6) FORECASTS
    Contains scripts to identify forecast objects  

7) OBSERVATIONS
    Contains scripts to identify observed objects 
    
8) HAGELSLAG
    Full Hagelslag package downloaded from GitHub


