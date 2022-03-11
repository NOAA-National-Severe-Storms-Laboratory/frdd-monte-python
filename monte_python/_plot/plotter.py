import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import multivariate_normal
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["white", "red", "blue", "green", "purple", 'gray'])


# Create fake storm data
def create_fake_storms(centers, nx=150):
    """ Create artifical storms """
    rng = range(nx)
    x,y = np.meshgrid(rng, rng)
    coords = np.dstack((x, y))
    
    storm_types = [[[50, 30], [30, 30]], 
                   [[40, 20], [20, 20]],
                   [[60, 20], [20, 50]],
                  ]
    
    random_state = np.random.RandomState(123)
    inds = random_state.choice(len(storm_types), size=len(centers), p=[0.4, 0.3, 0.3])
    
    data = 0 
    for ind, (i, j) in zip(inds, centers):
        data += (1e4 * multivariate_normal(mean=[i, j], cov=np.array(storm_types[ind])).pdf(coords))

    data = np.ma.masked_where(data<10., data)    
        
    return data, x, y     


def plot_fake_storms(x,y,data, ax=None, colorbar=True):
    """ Plot fake storms """
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4), dpi=150, facecolor='w', edgecolor='k')  
        
    c = ax.pcolormesh(x, y, data, cmap='jet', vmin=20, vmax=75)
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    if colorbar:
        plt.colorbar(c, label='Fake Reflectivity', ax=ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.5, ls='dashed')
    
    return ax 


def label_centroid(x, y, ax, object_props):
    """Place object label on object's centroid"""
    for region in object_props:
        x_cent,y_cent = region.centroid
        x_cent=int(x_cent)
        y_cent=int(y_cent)
        xx, yy = x[x_cent,y_cent], y[x_cent,y_cent]
        fontsize = 6.5 if region.label >= 10 else 8
        ax.text(xx,yy,
                    region.label,
                    fontsize=fontsize,
                    ha='center',
                    va='center',
                    color = 'k'
                    )    
    
def plot_storm_labels(x, y, labels, label_props, ax=None):
    """ Plot Storm Labels """
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 4), dpi=150, facecolor='w', edgecolor='k')
    
    labels = np.ma.masked_where(labels==0, labels)
    c = ax.pcolormesh(x, y, labels, cmap='tab20', vmin=1, vmax=np.max(labels))
    ax.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.5, ls='dashed')
    
    label_centroid(x, y, ax, label_props) 
    
    return ax     
    