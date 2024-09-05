from tffmodel.types.HeterogeneousSparseArray import HeterogeneousSparseArray

import numpy as np
import scipy

class SparseGradient(HeterogeneousSparseArray):
    @classmethod
    def sparsifyLayerwiseTopK(self_class, layer_gradients, k):
        def getTopKIndices(matrix, k):
            if(k > len(matrix.flatten())):
                k = len(matrix.flatten())
            sorted_idx = np.argpartition(matrix.flatten(), k)[-k:]
            sorted_idx = list(map(lambda idx: (idx // matrix.shape[0], idx % matrix.shape[0]), sorted_idx))
            return sorted_idx

        layerwise_topk_indices = [getTopKIndices(lg, k) for lg in layer_gradients]
        layerwise_topk_values = [[lg[idx] for idx in lti]
            for lg, lti in zip(layer_gradients, layerwise_topk_indices)]
        layerwise_topk_rowcols = [list(zip(*lti)) for lti in layerwise_topk_indices]
        sparse_layer_gradients = [scipy.sparse.csr_matrix((ltv, ltrc), shape=lg.shape)
            for ltv, ltrc, lg in zip(layerwise_topk_values, layerwise_topk_rowcols, layer_gradients)]
        return self_class(sparse_layer_gradients)
