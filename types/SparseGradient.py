from tffmodel.types.HeterogeneousSparseArray import HeterogeneousSparseArray

import math
import numpy as np
import scipy

class SparseGradient(HeterogeneousSparseArray):
    @classmethod
    def sparsifyLayerwiseTopK(self_class, gradient, k):
        def getTopKIndices(arr, k):
            if(k > len(arr)):
                k = len(arr)
            sorted_idx = np.argpartition(arr, len(arr)-k)[-k:]
            return sorted_idx

        layerwise_topk_indices = [getTopKIndices(lg.flatten(), k) for lg in gradient]

        masks = [np.zeros(lg.size) for lg in gradient]
        for idx, lti in enumerate(layerwise_topk_indices):
            masks[idx][lti] = 1
        masked_arrays = [lg * m.reshape(lg.shape) for lg, m in zip(gradient, masks)]

        sparse_gradient = self_class(masked_arrays)
        return sparse_gradient
