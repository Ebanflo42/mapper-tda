from scipy.spatial.distance import cdist, pdist
import pandas as pd
import numpy as np

"""
Filter functions take data, plus some additional paramaters,
and return a function which takes an index from the data and outputs a real number.
"""

def eccentricity_p(data, p):

    covariance = [v if v > 0 else 1. for v in np.var(data, axis=0)]

    def _fin_ecc(x, data):
        num = np.power(cdist([x], data, metric='seuclidean', V=covariance)[0], p)
        result = np.sum(num) / len(data)
        return np.power(result, 1. / p)

    def _inf_ecc(x, data):
        return np.max(cdist([x], data, metric='seuclidean', V=covariance)[0])

    if p == 'inf':
        return (lambda i: _inf_ecc(data[i], data))
    else:
        return (lambda i: _fin_ecc(data[i], data))


def axis_proj(data, coordinate):
    return (lambda i: data[i][coordinate])
