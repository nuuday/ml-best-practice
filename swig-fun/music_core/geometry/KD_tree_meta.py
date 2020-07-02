from .fast_map import FastMap
from scipy.spatial import cKDTree
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KDTreeMeta:
    """
     KD-tree Wrapper
    """

    def __init__(self, meta, s):
        """
        Constructs Approximate nearest neighbor search object given
        a set of feature samples.

        :param s: Feature samples
        :param meta: Meta data
        """
        logger.debug("Building on embedded space")
        self.kd_tree = cKDTree(s)
        self._meta = meta

    def __call__(self, x, n=None):
        """
        Find the n nearest neighhor to x.

        :param x: Feature sample
        :param n: The number of "nearest" neighbors
        :return: The "nearest" neighbors
        """

        n = n if n is not None else len(self._meta)//100

        # query to kd-tree n nearest neighbor candidates
        _, inrs = self.kd_tree.query(x, n)

        return [self._meta[i] for i in inrs]
