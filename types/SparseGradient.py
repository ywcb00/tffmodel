from tffmodel.types.HeterogeneousSparseArray import HeterogeneousSparseArray

from enum import Enum
import math
import numpy as np
import scipy

class SparsificationType(Enum):
    LAYERWISE_TOPK = 1
    LAYERWISE_PERCENTAGE = 2

# returns the indices of the K elements with highest absolute value
def getTopKIndices(arr, k):
    if(k > len(arr)):
        k = len(arr)
    sorted_idx = np.argpartition(np.absolute(arr), len(arr)-k)[-k:]
    return sorted_idx

class SparseGradient(HeterogeneousSparseArray):
    @classmethod
    def sparsifyGradient(self_class, gradient, config):
        match config["sparsification_type"]:
            case SparsificationType.LAYERWISE_TOPK:
                return self_class.sparsifyLayerwiseTopK(gradient, config["sparsification_k"])
            case SparsificationType.LAYERWISE_PERCENTAGE:
                return self_class.sparsifyLayerwisePercentage(gradient, config["sparsification_percentage"])
            case _:
                raise NotImplementedError

    # Keep only the K highest values per layer, set the others to zero
    @classmethod
    def sparsifyLayerwiseTopK(self_class, gradient, k):
        layerwise_topk_indices = [getTopKIndices(lg.flatten(), k) for lg in gradient]

        masks = [np.zeros(lg.size) for lg in gradient]
        for idx, lti in enumerate(layerwise_topk_indices):
            masks[idx][lti] = 1
        masked_arrays = [lg * m.reshape(lg.shape) for lg, m in zip(gradient, masks)]

        sparse_gradient = self_class(masked_arrays)
        return sparse_gradient

    # Keep the specified percentage of highest values per layer, set the others to zero
    @classmethod
    def sparsifyLayerwisePercentage(self_class, gradient, percentage):
        layerwise_percentage_indices = [getTopKIndices(
            lg.flatten(), math.ceil(lg.size*percentage)) for lg in gradient]

        masks = [np.zeros(lg.size) for lg in gradient]
        for idx, lpi in enumerate(layerwise_percentage_indices):
            masks[idx][lpi] = 1
        masked_arrays = [lg * m.reshape(lg.shape) for lg, m in zip(gradient, masks)]

        sparse_gradient = self_class(masked_arrays)
        return sparse_gradient
