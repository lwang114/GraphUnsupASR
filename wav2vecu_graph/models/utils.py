import random
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import find_peaks


def replicate_first_k_frames(x, k, dim):
    return torch.cat([x.index_select(dim=dim, index=torch.LongTensor([0] * k).to(x.device)), x], dim=dim)

def detect_peaks(x, lengths, prominence=0.1, width=None, distance=None):                      
    """detect peaks of next_frame_classifier                       
    
    Arguments:                                                     
        x {Tensor} -- batch of confidence per time                 
    """                                                            
    out = []

    for xi, li in zip(x, lengths):                                 
        if type(xi) == torch.Tensor:                               
            xi = xi.cpu().detach().numpy()                         
        xi = xi[:li]  # shorten to actual length                   
        xmin, xmax = xi.min(), xi.max()                            
        xi = (xi - xmin) / (xmax - xmin)
        peaks, _ = find_peaks(xi, prominence=prominence, width=width, distance=distance)
        
        if len(peaks) == 0:
            peaks = np.array([len(xi)-1])

        out.append(peaks)
        
    return out

def max_min_norm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x
