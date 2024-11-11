from tffmodel.types.HeterogeneousArray import HeterogeneousArray
from tffmodel.types.MultiDimSparseArray import MultiDimSparseArray

import numpy as np
import sparse

class HeterogeneousDenseArray(HeterogeneousArray):
    def __init__(self, data_arrays):
        if(not isinstance(data_arrays, np.ndarray)):
            data_arrays = [da.todense() if isinstance(da, sparse.COO) else da
                for da in data_arrays]
        super().__init__(np.array(data_arrays, dtype=object))

    @classmethod
    def getZero(self_class, shape_layer_arrays):
        layer_arrays = [np.zeros(sla.shape) for sla in shape_layer_arrays]
        return self_class(layer_arrays)

    def get(self):
        return self._data

    def getFlattened(self):
       flattened_array = np.concatenate([da.flatten() for da in self._data], axis=0)
       return flattened_array

    @classmethod
    def fromFlattened(self_class, flattened_array, shape_layer_arrays):
        layer_sizes = shape_layer_arrays.getSizes()
        indices = [0, *np.cumsum(layer_sizes)]
        layer_arrays = [flattened_array[indices[counter]:indices[counter+1]]
            for counter in range(len(indices)-1)]
        layer_arrays = [layer_arr.reshape(layer_shape) for layer_arr, layer_shape in zip(layer_arrays, shape_layer_arrays.getShapes())]
        return self_class(layer_arrays)

    def serialize(self):
        # NOTE: we transfer the arrays as float32 albeit it is float64
        # TODO: find a more efficient way to serialize the arrays
        serialized_array = []
        for layer_array in self._data:
            serialized_array.extend([np.float32(layer_array).tobytes(),
                np.int32(layer_array.shape).tobytes()])
        return serialized_array

    @classmethod
    def deserialize(self_class, serialized_array):
        data_arrays_and_shapes = [(np.frombuffer(serialized_array[idx], dtype=np.float32),
                np.frombuffer(serialized_array[idx+1], dtype=np.int32))
            for idx in range(0, len(serialized_array), 2)]
        data_arrays = [layer_array.reshape(layer_shape)
            for layer_array, layer_shape in data_arrays_and_shapes]
        return self_class(data_arrays)

    @classmethod
    def add_primitive(self_class, lhs, rhs):
        if(isinstance(lhs, HeterogeneousDenseArray) and isinstance(rhs, HeterogeneousDenseArray)):
            res = np.add(lhs._data, rhs._data)
        elif(isinstance(lhs, HeterogeneousArray) and isinstance(rhs, HeterogeneousArray)):
            res = [lhsdat + rhsdat for lhsdat, rhsdat in zip(lhs, rhs)]
        else:
            res = np.add(lhs._data, rhs)
        return res
    @classmethod
    def add_operation(self_class, lhs, rhs):
        res = self_class.add_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def sub_primitive(self_class, lhs, rhs):
        if(isinstance(lhs, HeterogeneousDenseArray) and isinstance(rhs, HeterogeneousDenseArray)):
            res = np.subtract(lhs._data, rhs._data)
        elif(isinstance(lhs, HeterogeneousArray) and isinstance(rhs, HeterogeneousArray)):
            res = [lhsdat - rhsdat for lhsdat, rhsdat in zip(lhs, rhs)]
        else:
            res = np.subtract(lhs._data, rhs)
        return res
    @classmethod
    def sub_operation(self_class, lhs, rhs):
        res = self_class.sub_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def mul_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousDenseArray)):
            res = np.multiply(lhs._data, rhs._data)
        else:
            res = np.multiply(lhs._data, rhs)
        return res
    @classmethod
    def mul_operation(self_class, lhs, rhs):
        if(not isinstance(rhs, HeterogeneousDenseArray)
            and isinstance(rhs, HeterogeneousArray)): # is sparse array
            return rhs.mul_operation(lhs, rhs)
        res = self_class.mul_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def div_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousDenseArray)):
            res = np.divide(lhs._data, rhs._data)
        else:
            res = np.divide(lhs._data, rhs)
        return res
    @classmethod
    def div_operation(self_class, lhs, rhs):
        res = self_class.div_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def eq_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousDenseArray)):
            res = np.all(np.vectorize(np.array_equal)(lhs._data, rhs._data))
            return res
        elif(isinstance(rhs, HeterogeneousArray)): # is sparse
            res = all([(lhsdat == rhsdat).all() for lhsdat, rhsdat in zip(lhs, rhs)])
        else:
            res = all([(lhsdat == rhs).all()for lhsdat in lhs])
