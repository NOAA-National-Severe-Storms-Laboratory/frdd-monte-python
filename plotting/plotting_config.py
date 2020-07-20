# Plotting config file
from wofs.util.news_e_plotting_cbook_v2 import cb_colors as wofs
line_colors = [ wofs.red8, wofs.green8, wofs.blue8, wofs.orange7, wofs.gray8, 'k' ]

colors_for_ml_models_dict = { 'RandomForest': wofs.red8,
                              'XGBoost': wofs.blue8, 
                              'LogisticRegression': wofs.green8,
                              'ensemble': wofs.gray8,
                              'CNN': wofs.gray8
                              } 

title_dict = {'matched_to_tornado_0km':'Tornadoes',
              'matched_to_severe_hail_0km':'Severe Hail',
              'matched_to_severe_wind_0km':'Severe Wind'
              }

corr_title_dict = {True: 'Correlated Features Removed',
                   False: 'All Features'
                  }


def colors_per_time( line_labels ):
    """ Get colors for a model at different times """
    colors = [wofs.red8, wofs.blue8, wofs.green8, wofs.gray8, wofs.orange8, wofs.purple8]
    colors_for_different_times = {l: c for c,l in zip(colors[:len(line_labels)], line_labels)}
    return colors_for_different_times

def get_line_labels( model_name, fcst_time_idx_set, duration=30):
    fcst_time_idx_set = [0,6,12]
    map_dict = {t: f'{(t*5)}-{(t*5)+duration} min' for t in fcst_time_idx_set} 
    line_labels = [f'{model_name} {map_dict[t]}' for t in fcst_time_idx_set]

    return line_labels


