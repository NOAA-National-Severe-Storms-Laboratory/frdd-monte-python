import numpy as np 


def replace_zeros(data): 
    return np.where(data==0, 1e-5, data)

class ObjectVerifier:
    """
    ObjectVerifier computes simple contingency table metrics based on matching 
    objects between prediction and target objects. 
    
    Attributes
    --------------
    object_matcher : An initialized MontePython.ObjectMatcher object. Used for matching 
    the target and prediction object fields. 
    
    """
    def __init__(self, object_matcher): 
        self._object_matcher = object_matcher 

        self.hits_ = 0
        self.false_alarms_ = 0
        self.misses_ = 0 
        
    def update_metrics(self, target, prediction): 
        """ Compute the hits, misses, and false alarms based on the object matching.
        
        TODO: This is not robust for non-one-to-one matching!!
        
        Parameters
        ------------
        target : 2d numpy.ndarray : The labelled target field 
        prediction : 2d numpy.ndarray: The labelled prediction field 
        
        """
        matched_pred, matched_tar, dists = self._object_matcher.match(prediction, target)
    
    
        n_total_pred_objs = len(np.unique(prediction)[1:]) #[1:] to ignore the 0 label
        n_total_tar_objs = len(np.unique(target)[1:])
    
        n_matched_pred = len(matched_pred)
        n_matched_tar = len(matched_tar)
    
        assert_txt = """Number of matched predictions exceed number of predictions objects"""
        assert n_total_pred_objs >= n_matched_pred , assert_txt
        
        self.hits_ += n_matched_pred 
        self.false_alarms_ += (n_total_pred_objs - n_matched_pred ) 
        self.misses_ += (n_total_tar_objs - n_matched_tar)
    
    def reset_metrics(self):
        """Resets the counts of hits, false alarms, and misses to zero."""
        self.hits_ = 0
        self.false_alarms_ = 0
        self.misses_ = 0
    
    @property
    def pod(self):
        """Hits / Hits + Misses"""
        return self.hits_ / (self.hits_ + self.misses_)
    
    @property
    def sr(self):
        """Hits / Hits + False Alarms"""
        return self.hits_ / (self.hits_ + self.false_alarms_)
    
    @property
    def csi(self):
        """Hits / Hits + Misses + False Alarms"""
        return self.hits_ / (self.hits_ + self.false_alarms_ + self.misses_)
    