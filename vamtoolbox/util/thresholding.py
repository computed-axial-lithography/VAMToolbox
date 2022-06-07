import numpy as np



def threshold(x,thresh):

    x_thresh = np.array(x >= thresh, dtype=np.bool)

    return x_thresh
