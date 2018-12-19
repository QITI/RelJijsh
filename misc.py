"""Summary
"""
import numpy as np

def diag_idcs(n,k):
    """Summary

    Args:
        n (TYPE): Description
        k (TYPE): Description

    Returns:
        TYPE: Description
    """
    i,j = np.diag_indices(n)
    if k < 0:
        return i[k:],i[:-k]
    if k > 0:
        return i[:-k],i[k:]
    else:
        return i,j

def intersection(x,y,exclusion):
    return [i for i in x if i in y and i not in exclusion]
