from tffmodel.types.HeterogeneousArray import HeterogeneousArray
from tffmodel.types.MultiDimSparseArray import MultiDimSparseArray

import numpy as np

class HeterogeneousSparseArray(HeterogeneousArray):
    def __init__(self, data_arrays):
        if(any(map(lambda darr: not isinstance(darr, MultiDimSparseArray), data_arrays))):
            data_arrays = [MultiDimSparseArray(darr) for darr in data_arrays]
        super().__init__(data_arrays)

    @classmethod
    def getZero(self_class, shape_layer_arrays):
        layer_arrays = [MultiDimSparseArray.zeros(sla.shape) for sla in shape_layer_arrays]
        return self_class(layer_arrays)

    def get(self):
        return np.array([darr.to_numpy() for darr in self._data], dtype=object)

    def getNNZ(self):
        nnz = [layer_array.nnz for layer_array in self._data]
        return nnz

    def serialize(self):
        # NOTE: the coordinates are serialized as 32-bit integers
        # create artificial list of with (coordinates, data, shape) entries
        serialized_array = []
        for layer_array in self._data:
            target_dtype = layer_array.dtype
            serialized_array.extend([np.int32(layer_array.coords).flatten().tobytes(),
                layer_array.data.astype(target_dtype).tobytes(),
                np.int32(layer_array.shape).tobytes(),
                target_dtype.str.encode("UTF-8")])
        return serialized_array

    @classmethod
    def deserialize(self_class, serialized_array):
        data_arrays = []
        for idx in range(0, len(serialized_array), 4):
            layer_shape = np.frombuffer(serialized_array[idx+2], dtype=np.int32)
            layer_coords = np.frombuffer(serialized_array[idx], dtype=np.int32)
            layer_coords = layer_coords.reshape(
                (len(layer_shape), len(layer_coords)//len(layer_shape)))
            layer_dtype = np.dtype(serialized_array[idx+3].decode("UTF-8"))
            layer_data = np.frombuffer(serialized_array[idx+1], dtype=layer_dtype)
            data_arrays.append(MultiDimSparseArray(
                coords=layer_coords, data=layer_data, shape=layer_shape))
        return self_class(data_arrays)

    def sparsify(self, mask):
        masked_data = [layer * m.reshape(layer.shape) for layer, m in zip(self._data, mask)]
        return self.__class__(masked_data)

    @classmethod
    def add_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousSparseArray)):
            res = [lhsdat + rhsdat for lhsdat, rhsdat in zip(lhs, rhs)]
        else:
            res = [lhsdat + rhs for lhsdat in lhs._data]
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
            res = [lhsdat - rhsdat for lhsdat, rhsdat in zip(lhs, rhs)]
        else:
            res = [lhsdat - rhs for lhsdat in lhs._data]
        return res
    @classmethod
    def sub_operation(self_class, lhs, rhs):
        if(not isinstance(rhs, HeterogeneousSparseArray)
            and isinstance(rhs, HeterogeneousArray)): # is dense array
            return rhs.sub_operation(lhs, rhs)
        res = self_class.sub_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def mul_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousArray)):
            res = [lhsdat * rhsdat for lhsdat, rhsdat in zip(lhs, rhs)]
        else:
            res = [lhsdat * rhs for lhsdat in lhs._data]
        return res
    @classmethod
    def mul_operation(self_class, lhs, rhs):
        res = self_class.mul_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def div_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousArray)):
            res = [lhsdat / rhsdat for lhsdat, rhsdat in zip(lhs, rhs)]
        else:
            res = [lhsdat / rhs for lhsdat in lhs._data]
        return res
    @classmethod
    def div_operation(self_class, lhs, rhs):
        res = self_class.div_primitive(lhs, rhs)
        return self_class(res)
    @classmethod
    def eq_primitive(self_class, lhs, rhs):
        if(isinstance(rhs, HeterogeneousArray)):
            res = all([(lhsdat == rhsdat).all() for lhsdat, rhsdat in zip(lhs, rhs)])
        else:
            res = all([(lhsdat == rhs).all() for lhsdat in lhs])
        return res
