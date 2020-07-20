import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.utils import resample
from numpy.random import uniform

class ContingencyTable:
    ''' Calculates the values of the contingency table.
    param: truths, True binary labels. shape = [n_samples] 
    param: forecasts, forecasts binary labels. shape = [n_samples]
    ContingencyTable calculates the components of the contigency table, but ignoring correct negatives. 
    Can use determinstic and probabilistic input.     
    '''
    def __init__( self, truths, forecasts ): 
        hits = np.sum( np.where(( truths == 1) & ( forecasts == 1 ), 1, 0 ) )
        false_alarms = np.sum( np.where(( truths == 0) & ( forecasts == 1 ), 1, 0 ) )
        misses = np.sum( np.where(( truths == 1) & ( forecasts == 0), 1, 0 ) )
        corr_negs = np.sum( np.where(( truths == 0) & ( forecasts == 0 ),  1, 0 ) ) 

        self.table = np.array( [ [hits, misses], [false_alarms, corr_negs]], dtype=float) 
        # Hit: self.table[0,0]
        # Miss: self.table[0,1]
        # False Alarms: self.table[1,0]
        # Corr Neg. : self.table[1,1] 

    def calc_pod(self):
        '''
        Probability of Detection (POD) or Hit Rate. 
        Formula: hits / hits + misses
        '''
        ##print self.table[0,0] / (self.table[0,0] + self.table[0,1])
        return self.table[0,0] / (self.table[0,0] + self.table[0,1])

    def calc_pofd(self):
        '''
        Probability of False Detection.
        Formula: false alarms / false alarms + correct negatives
        '''
        return self.table[1,0] / (self.table[1,0] + self.table[1,1])

    def calc_sr(self):
        '''
        Success Ratio (1 - FAR).
        Formula: hits / (hits+false alarms)
        '''
        if self.table[0,0] + self.table[1,0] == 0.0:
            return 1.
        else:
            return self.table[0,0] / (self.table[0,0] + self.table[1,0])

    @staticmethod
    def calc_bias(pod, sr):
        '''
        Frequency Bias.
        Formula: POD / SR ; (hits + misses) / (hits + false alarms)  
        '''
        sr[np.where(sr==0)] = 1e-5
        return pod / sr 

    @staticmethod
    def calc_csi(pod, sr):
        '''
        Critical Success Index.
        Formula: Hits / ( Hits+Misses+FalseAlarms)
        '''
        sr[np.where(sr==0)] = 1e-5
        pod[np.where(pod==0)] = 1e-5
        return 1. /((1./sr) + (1./pod) - 1.)

    @staticmethod
    def determine_contingency_table_components( observed_labels, forecast_labels, matched_observed_labels, matched_forecast_labels, forecast_probs = None ):
        """ Determines whether forecast labels are false alarms or hits and whether 
                observed labels are misses or hits 
        Args: 
            observed_labels, 2D array with labels for observed objects 
            forecast_labels, 2D array with labels for forecast objects
            matched_observed_labels, list of matched observed object labels in observed_labels
            matched_forecast_labels, list of matched forecast object labels in forecast_labels 
            forecast_probs, 2D array with forecast probabilities ( from which probabiliity forecast objects were generated from )   
        Returns: 
            Lists of contingency table outcome for the forecast and observations 
        """
        forecast_outcome = [ ]
        observed_outcome = [ ]

        if forecast_probs is not None: 
            prob_object_value = calc_prescribed_prob_value( forecast_probs, forecast_labels )

        all_forecast_labels = np.unique( forecast_labels )[1:]
        
        if len(np.shape(observed_labels)) == 1:
            all_observed_labels = observed_labels
        else:
            all_observed_labels = np.unique( observed_labels )[1:]

        # False alarms are unmatched forecast labels
        for forecast_label in all_forecast_labels:
            if forecast_label not in matched_forecast_labels:
                if forecast_probs is not None:
                    forecast_outcome.append( prob_object_value[forecast_label] )
                else:
                    forecast_outcome.append( 1 )
                observed_outcome.append( 0 )

        # Misses are unmatched observation labels
        for observed_label in all_observed_labels:
            if observed_label not in matched_observed_labels:
                if forecast_probs is not None:
                    # Don't want to append zero since when CPT == 0 is applied 
                    # misses would apply as hits 
                    forecast_outcome.append( -1 )
                else:
                    forecast_outcome.append( 0 )
                observed_outcome.append( 1 )

        # Hits are all labels in matched forecast labels
        for forecast_label in matched_forecast_labels:
            if forecast_probs is not None:
                forecast_outcome.append( prob_object_value[forecast_label] )
            else:
                forecast_outcome.append( 1 )
            observed_outcome.append( 1 )

        return forecast_outcome, observed_outcome

def calc_prescribed_prob_value( forecast_probs, forecast_labels ):
    """ 
    Returns dictionary where keys are the object labels and value is the 
    maximum probability value. 
    """
    prob_object_value = { }
    for label in np.unique(forecast_labels)[1:]:
        prob_object_value[label] = np.amax( forecast_probs[np.where( forecast_labels == label )] )

    return prob_object_value # { 1: 0.9, 2: 0.45, 3: 0.67 } 

class Metrics:
    @staticmethod
    def performance_curve( forecasts, truths, bins=np.arange(0, 1, 0.01), deterministic=False, roc_curve=False ): 
        ''' 
        Generates the POD and SR for a series of probability thresholds 
        to produce performance diagram (Roebber 2009) curves
        '''
        if deterministic:
            table = ContingencyTable( truths, forecasts )
            pod = table.calc_pod( )
            sr = table.calc_sr( )
        else:    
            pod = np.zeros((bins.shape))
            sr = np.zeros((bins.shape))
            if roc_curve:
                pofd = np.zeros((bins.shape))
            for i, p in enumerate( bins ): 
                p = round(p,5)
                binary_fcst = np.where( np.round(forecasts,10) >= p, 1, 0 ) 
                table = ContingencyTable( truths.astype(int), binary_fcst.astype(int) )
                pod[i] = table.calc_pod( )
                sr[i] = table.calc_sr( )
                if roc_curve:
                    pofd[i] = table.calc_pofd()

            del table
        if roc_curve:
            return pod, sr, pofd 
        else:
            return pod, sr
        
    @staticmethod
    def reliability_curve( forecasts, truths, bins=np.arange(0, 1+1./18., 1./18.)):
        '''
        Generate reliability (calibration) curve.
        '''
        mean_fcst_probs = np.zeros(( len(bins)-1))
        event_frequency = np.zeros(( len(bins)-1))

        for i, p in enumerate( bins[:-1] ):
            lower_bound = round( bins[i], 5)
            upper_bound = round( bins[i+1],5)
            bin_indices = np.where((forecasts>=lower_bound)&(forecasts<upper_bound))
           
            
            if len( np.ravel(bin_indices) ) > 0:  
                mean_fcst_probs[i] = np.mean(forecasts[bin_indices])         
                pos_frequency = np.sum(truths[bin_indices])
                event_frequency[i] = float(pos_frequency) / len( np.ravel(bin_indices) ) 
            else:
                # No forecasts in this particular bin
                mean_fcst_probs[i] = np.nan
                event_frequency[i] = np.nan 
        
        return mean_fcst_probs, event_frequency
    

def reliability_uncertainty( X, Y, N_boot = 1000, bins=np.arange(0, 1+1./18., 1./18.) ):
    '''
    Calculates the uncertainty of the event frequency based on Brocker and Smith (WAF, 2007)
    '''
    event_freq_set = [ ]
    for i in range( N_boot ):
        Z     = uniform( size = len(X) )
        X_hat = resample( X )
        Y_hat = np.where( Z < X_hat, 1, 0 )
        _, event_freq = Metrics.reliability_curve( X_hat, Y_hat, bins=bins )
        event_freq_set.append( event_freq )

    return np.array( event_freq_set )


def _get_binary_xentropy(target_values, forecast_probabilities):
    """Computes binary cross-entropy.

    This function satisfies the requirements for `cost_function` in the input to
    `run_permutation_test`.

    E = number of examples

    :param: target_values: length-E numpy array of target values (integer class
        labels).
    :param: forecast_probabilities: length-E numpy array with predicted
        probabilities of positive class (target value = 1).
    :return: cross_entropy: Cross-entropy.
    """
    MIN_PROBABILITY = 1e-15
    MAX_PROBABILITY = 1. - MIN_PROBABILITY
    forecast_probabilities[
        forecast_probabilities < MIN_PROBABILITY] = MIN_PROBABILITY
    forecast_probabilities[
        forecast_probabilities > MAX_PROBABILITY] = MAX_PROBABILITY

    return -1 * np.nanmean(
        target_values * np.log2(forecast_probabilities) +
        (1 - target_values) * np.log2(1 - forecast_probabilities))

def brier_skill_score(target_values, forecast_probabilities):
    climo = np.mean((target_values - np.mean(target_values))**2)
    return 1.0 - brier_score_loss(target_values, forecast_probabilities) / climo






