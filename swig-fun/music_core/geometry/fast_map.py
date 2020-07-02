from random import sample
import numpy as np
import json

from sklearn.utils.extmath import randomized_svd
from .divergence import NormalDistribution

import logging
logger = logging.getLogger(__name__)

import base64

class FastMap:
    """
         A modified FastMap implementation as described in "Indexing Content-Based
         Music Similarity Models for Fast Retrieval in Massive Databases"
     """

    def __init__(self, samples, distance_function=NormalDistribution.js, k_pivot=10, k_components=None):
        """
        Construct a modified FastMap.

        :param samples: Feature samples
        :param distance_function: The metric or approximate metric to measure similarity
        :param k_pivot: The number of pivot elements
        :param k_components: The number of components
        """

        self._d = distance_function
        self._pivot(samples, k_pivot)
        self._f_mean=np.zeros(k_pivot)
        self._components=np.identity(k_pivot)
        self._reduce = lambda x: x

        if k_components is not None and k_components < k_pivot:
            f = [self(x) for x in samples]
            self._f_mean = np.array(f).mean(axis=0)
            f -= self._f_mean

            u, self._sigma, v = randomized_svd(f, k_components)
            self._components = v

            # Calculate explained variance & explained variance ratio
            f_transformed = u * self._sigma
            self.explained_variance_ = exp_var = np.var(f_transformed, axis=0)
            full_var = np.var(f, axis=0).sum()
            self.explained_variance_ratio_ = exp_var / full_var

            self.reduce = lambda x: self._components.dot(x-self._f_mean)

    def _dist(self, samples, y):
        """
        Computes the distance of a feature against a sample of features

        :param samples: Feature sample
        :param y: The feature
        :return:
        """
        return np.array([self._d(x, y) for x in samples])

    def _pivot(self, samples, k):
        """
        Computes k pivot elements from a sample of features

        :param samples: Feature samples
        :param k: The number of pivot elements
        :return:
        """

        # the pivot elements
        self._E = []

        # iterate a random set of k features
        for x0 in sample(samples, k):

            # find median distanced feature from sample
            # and use it as first pivot element
            d = self._dist(samples, x0)
            i1 = np.argsort(d)[len(d) // 2]
            x1 = samples[i1]

            # find median distanced feature from first pivot element
            # and use it as second pivot element
            d = self._dist(samples, x1)
            i2 = np.argsort(d)[len(d) // 2]

            # save pivot elements
            self._E.append((x1, samples[i2], d[i2]))

    def __call__(self, x):
        """
        Embed feature in lower dimensional space
        :param x: The feature
        :return: The embedding
        """
        return self._reduce(np.array([(self._d(x1, x)**2 + d12**2 - self._d(x2, x)**2)/(2*d12)
                         for x1, x2, d12 in self._E]))

    def __dict__(self):
        return {
            '_E': self._E,
            '_components': base64.b64encode(self._components.tobytes()).decode('ascii'),
            '_f_mean': base64.b64encode(self._f_mean.tobytes()).decode('ascii'),
            '_d': self._d.__name__
        }

    @staticmethod
    def from_json(json_string):
        json_data = json.loads(json_string)

        _f_mean = base64.b64decode(json_data["_f_mean"])
        _f_mean = np.frombuffer(_f_mean, dtype=np.float)

        _components = base64.b64decode(json_data["_components"])
        _components = np.frombuffer(_components, dtype=np.float).reshape([-1, len(_f_mean)])

        _d = getattr(NormalDistribution,json_data["_d"])


        _E = [(NormalDistribution(**e[0]), NormalDistribution(**e[1]), e[2]) for e in json_data["_E"]]
        
        fast_map = object.__new__(FastMap)
        
        fast_map._E = _E
        fast_map._f_mean = _f_mean
        fast_map._components = _components
        fast_map._d = _d

        fast_map._reduce = lambda x: _components.dot(x-_f_mean)

        return fast_map


