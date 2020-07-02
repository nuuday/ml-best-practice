from .fast_map import FastMap
from scipy.spatial import cKDTree
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ApproximateNearestNeighbor:
    """
    An approximate nearest neighborhood implementation based
    on FastMap and KD-tree
    """

    def __init__(self, s, distance_function, fast_map, filter_size=0.1):
        """
        Constructs Approximate nearest neighbor search object given
        a set of feature samples.

        :param s: Feature samples
        :param distance_function: The metric or approximate metric to measure similarity
        :param k: The number of pivot elements
        :param kout: The number of PCA components - if None just use fast map embedding directly
        :param filter_size: The filter size relative to the total number of samples
        """
        self.fast_map = fast_map if fast_map is not None else FastMap(s, distance_function,  100, 10)

        self._d = distance_function
        self._s = s
        self.filter_size = int(len(s)*filter_size)
        self.fast_map = fast_map

        f = [self.fast_map(x) for x in s]

        logger.debug("Building on embedded space")
        self.kd_tree = cKDTree(f)

    def __call__(self, x, n=100, filter_size=None):
        """
        Find the n nearest neighhor to x.

        :param x: Feature sample
        :param n: The number of "nearest" neighbors
        :return: The "nearest" neighbors
        """
        # embeds feature in lower dimensional space
        f = self.fast_map(x)

        # query to kd-tree 5*N nearest neighbor candidates
        filter_size = filter_size if filter_size is not None else self.filter_size
        _, inrs = self.kd_tree.query(f, filter_size)

        # computes distance to nearest neighbor candidates
        dist = [self._d(x, self._s[i]) for i in inrs]

        # computes the n nearest neighbors from candidate set
        ind = np.argsort(dist)[:n]

        return [self._s[i] for i in inrs[ind]]
