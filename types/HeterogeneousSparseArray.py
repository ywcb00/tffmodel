from tffmodel.types.HeterogeneousArray import HeterogeneousArray

import numpy as np
import scipy

class HeterogeneousSparseArray(HeterogeneousArray):
    def __init__(self, data_arrays):
        self._data = np.array(data_arrays, dtype=object)

    @classmethod
    def getZero(self_class, shape_layer_arrays):
        layer_arrays = [scipy.sparse.csr_matrix(([], [[], []]), shape=sla.shape)
            for sla in shape_layer_arrays]
        return self_class(layer_arrays)

    def serialize(self):
        # create artificial list of with (data, indices, indptr) entries
        dii_list = []
        for layer_array in self._data:
            dii_list.extend([layer_array.data, layer_array.indices, layer_array.indptr])
        serialized_array = [np.float32(arr).tobytes() for arr in dii_list]
        return serialized_array

    @classmethod
    def deserialize(self_class, serialized_array, shape_arrays):
        dii_list = [np.frombuffer(arr, dtype=np.float32) for arr in serialized_array]
        dii_list = [(dii_list[idx], dii_list[idx+1], dii_list[idx+2])
            for idx in range(0, len(dii_list), 3)]
        data_arrays = [scipy.sparse.csr_matrix(dii_entry, shape=shape_arrays[idx].shape)
            for idx, dii_entry in enumerate(dii_list)]
        return self_class(data_arrays)

    @classmethod
    def add_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousSparseArray)):
            res = np.add(lhs._data, rhs._data)
        else:
            res = np.add(lhs._data, rhs)
        return res
    @classmethod
    def add_operation(self_class, lhs, rhs):
        if(not isinstance(rhs, HeterogeneousSparseArray)
            and isinstance(rhs, HeterogeneousArray)): # dense array
            return rhs.add_operation(lhs, rhs)
        res = self_class.add_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def sub_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousSparseArray)):
            res = np.subtract(lhs._data, rhs._data)
        else:
            res = np.subtract(lhs._data, rhs)
        return res
    @classmethod
    def sub_operation(self_class, lhs, rhs):
        if(not isinstance(rhs, HeterogeneousSparseArray)
            and isinstance(rhs, HeterogeneousArray)):
            return rhs.sub_operation(lhs, rhs)
        res = self_class.sub_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def mul_primitive(self_class, lhs, rhs):
        if(isinstance(lhs, HeterogeneousSparseArray) and isinstance(rhs, HeterogeneousSparseArray)):
            res = np.multiply(lhs._data, rhs._data)
        elif((not isinstance(lhs, HeterogeneousSparseArray)
                and isinstance(lhs, HeterogeneousArray))
            or (not isinstance(rhs, HeterogeneousSparseArray)
                and isinstance(rhs, HeterogeneousArray))): # one array is dense
            res = np.array([lshda * rshda for lshda, rshda in zip(lhs, rhs)], dype=object)
        else:
            res = np.multiply(lhs._data, rhs)
        return res
    @classmethod
    def mul_operation(self_class, lhs, rhs):
        res = self_class.mul_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def div_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousSparseArray)):
            res = np.divide(lhs._data, rhs._data)
        elif(not isinstance(rhs, HeterogeneousSparseArray)
            and isinstance(rhs, HeterogeneousArray)):
            res = np.array([lhsda / rhsda for lhsda, rhsda in zip(lhs, rhs)], dtype=object)
        else:
            res = np.divide(lhs._data, rhs)
        return res
    @classmethod
    def div_operation(self_class, lhs, rhs):
        res = self_class.div_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def eq_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousSparseArray)):
            res = np.all(np.vectorize(np.array_equal)(lhs._data, rhs._data))
            return res
        else:
            return NotImplemented
