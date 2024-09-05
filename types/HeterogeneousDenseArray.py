from tffmodel.types.HeterogeneousArray import HeterogeneousArray

import numpy as np

class HeterogeneousDenseArray(HeterogeneousArray):
    def __init__(self, data_arrays, shapes=None):
        if(shapes == None):
            shapes = [np.array(layer_array.shape) for layer_array in data_arrays]
        data_arrays = [layer_array.flatten() for layer_array in data_arrays]
        super().__init__(data_arrays, shapes)

    @classmethod
    def getZero(self_class, shape_layer_arrays):
        layer_arrays = [np.zeros(sla.shape) for sla in shape_layer_arrays]
        return self_class(layer_arrays)

    def get(self):
        shaped_layer_arrays = np.array(
            [arr.reshape(shape) for arr, shape in zip(self._data, self._shapes)],
            dtype=object)
        return shaped_layer_arrays

    def serialize(self):
        # NOTE: we transfer the arrays as float32 albeit it is float64
        # TODO: find a more efficient way to serialize the arrays
        serialized_array = []
        for layer_array, layer_shape in zip(self._data, self._shapes):
            serialized_array.extend([np.float32(layer_array).tobytes(),
                np.int32(layer_shape).tobytes()])
        return serialized_array

    @classmethod
    def deserialize(self_class, serialized_array):
        deserialized_arrays = [(np.frombuffer(serialized_array[idx], dtype=np.float32),
                np.frombuffer(serialized_array[idx+1], dtype=np.int32))
            for idx in range(0, len(serialized_array), 2)]
        data_arrays, shapes = list(zip(*deserialized_arrays))
        data_arrays = list(data_arrays)
        shapes = list(shapes)
        return self_class(data_arrays, shapes)

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
        return self_class(res, lhs.getShapes())
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
        return self_class(res, lhs.getShapes())
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
        return self_class(res, lhs.getShapes())
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
        return self_class(res, lhs.getShapes())
    @classmethod
    def eq_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousDenseArray)):
            res = np.all(np.vectorize(np.array_equal)(lhs._data, rhs._data))
            return res
        else:
            return NotImplemented
