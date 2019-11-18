import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
import mapper as mp


def eccentricity_p(p, data):

    covariance = [v if v > 0 else 1. for v in np.var(data, axis=0)]

    def _fin_ecc(x, data):
        num = np.power(cdist(x, data, metric='seuclidean', V=covariance)[0], p)
        result = np.sum(num) / len(data)
        return np.power(result, 1. / p)

    def _inf_ecc(x, data):
        return np.max(cdist(x, data, metric='seuclidean', V=covariance)[0])

    if p == 'inf':
        return lambda x: _inf_ecc(x, data)
    else:
        return lambda x: _fin_ecc(x, data)


def AxisProj(coordinate):
    return lambda x: x[coordinate]
