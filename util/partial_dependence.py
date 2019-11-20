import numpy as np

def partial_dependence(df, model, variable_idx, **kwargs):
    '''
    Calculate the partial dependence.
    # Friedman, J., 2001: Greedy function approximation: a gradient boosting machine.Annals of Statistics,29 (5), 1189–1232.
    ##########################################################################
    Partial dependence plots fix a value for one or more predictors
    # for examples, passing these new data through a trained model, 
    # and then averaging the resulting predictions. After repeating this process
    # for a range of values of X*, regions of non-zero slope indicates that
    # where the ML model is sensitive to X* (McGovern et al. 2019). Only disadvantage is
    # that PDP do not account for non-linear interactions between X and the other predictors.
    #########################################################################
    '''
    all_values = df[:, variable_idx]
    variable_range = np.linspace(all_values.min(), all_values.max(), num = 10 )
    variable_range_unnormalized = (variable_range * kwargs['std'][variable_idx]) + kwargs['mean'][variable_idx]
    all_values = (all_values * kwargs['std'][variable_idx]) + kwargs['mean'][variable_idx]

    pdp_values = np.zeros(( variable_range.shape ))
    for i, value in enumerate(variable_range):
        copy_df = df.copy()
        copy_df[:, variable_idx] = np.ones(copy_df.shape[0])*value
        predictions = model.predict( copy_df )
        pdp_values[i] = np.mean(predictions)

    return pdp_values, variable_range_unnormalized, all_values
