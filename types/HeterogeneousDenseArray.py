from tffmodel.types.HeterogeneousArray import HeterogeneousArray

import numpy as np

class HeterogeneousDenseArray(HeterogeneousArray):
    def __init__(self, data_arrays):
        self._data = np.array(data_arrays, dtype=object)

    @classmethod
    def getZero(self_class, shape_layer_arrays):
        layer_arrays = [np.zeros(sla.shape) for sla in shape_layer_arrays]
        return self_class(layer_arrays)

    def serialize(self):
        # NOTE: we transfer the arrays as float32 albeit it is float64
        # TODO: find a more efficient way to serialize the arrays
        serialized_array = [np.float32(layer_array).tobytes() for layer_array in self._data]
        return serialized_array

    @classmethod
    def deserialize(self_class, serialized_array, shape_arrays):
        data_arrays = [np.frombuffer(layer_array, dtype=np.float32)
                .reshape(shape_arrays[idx].shape)
            for idx, layer_array in enumerate(serialized_array)]
        return self_class(data_arrays)

    @classmethod
    def add_primitive(self_class, lhs, rhs):
        if(isinstance(lhs, HeterogeneousDenseArray) and isinstance(rhs, HeterogeneousDenseArray)):
            res = np.add(lhs._data, rhs._data)
        elif(isinstance(lhs, HeterogeneousArray) and isinstance(rhs, HeterogeneousArray)):
            res = np.array([lhsda + rhsda for lhsda, rhsda in zip(lhs, rhs)], dtype=object)
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
            res = np.array([lhsda - rhsda for lhsda, rhsda in zip(lhs, rhs)], dtype=object)
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
        else:
            return NotImplemented
