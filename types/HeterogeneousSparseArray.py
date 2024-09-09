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
        # create artificial list of with (coordinates, data, shape) entries
        serialized_array = []
        for layer_array in self._data:
            serialized_array.extend([np.int32(layer_array.coords).flatten().tobytes(),
                np.float32(layer_array.data).tobytes(),
                np.int32(layer_array.shape).tobytes()])
        return serialized_array

    @classmethod
    def deserialize(self_class, serialized_array):
        coords_data_and_shapes = [(np.frombuffer(serialized_array[idx], dtype=np.int32),
                np.frombuffer(serialized_array[idx+1], dtype=np.float32),
                np.frombuffer(serialized_array[idx+2], dtype=np.int32))
            for idx in range(0, len(serialized_array), 3)]
        data_arrays = []
        for coords, data, shape in coords_data_and_shapes:
            coords = coords.reshape((len(shape), len(coords)//len(shape)))
            data_arrays.append(MultiDimSparseArray(coords=coords, data=data, shape=shape))
        return self_class(data_arrays)

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
