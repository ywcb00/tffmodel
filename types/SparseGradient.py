from tffmodel.types.HeterogeneousSparseArray import HeterogeneousSparseArray

import numpy as np
import scipy

class SparseGradient(HeterogeneousSparseArray):
    @classmethod
    def sparsifyLayerwiseTopK(self_class, layer_gradients, k):
        def resolveFlattenedIndex(idx, shape):
            if(len(shape) == 1):
                return [idx]
            res_idx = (*resolveFlattenedIndex(idx//shape[-1], shape[:-1]), idx % shape[-1])
            return res_idx
        def getTopKIndices(matrix, k):
            if(k > len(matrix.flatten())):
                k = len(matrix.flatten())
            sorted_idx = np.argpartition(matrix.flatten(), len(matrix.flatten())-k)[-k:]
            sorted_idx = list(map(lambda idx: resolveFlattenedIndex(idx, matrix.shape), sorted_idx))
            return sorted_idx

        layerwise_topk_indices = [getTopKIndices(lg, k) for lg in layer_gradients]
        layerwise_topk_values = [[lg[idx] for idx in lti]
            for lg, lti in zip(layer_gradients, layerwise_topk_indices)]
        layerwise_topk_rowcols = [list(zip(*lti)) for lti in layerwise_topk_indices]
        sparse_layer_gradients = [scipy.sparse.csr_matrix((ltv, ltrc), shape=lg.shape)
            for ltv, ltrc, lg in zip(layerwise_topk_values, layerwise_topk_rowcols, layer_gradients)]
        return self_class(sparse_layer_gradients)
