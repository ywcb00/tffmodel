from tffmodel.types.HeterogeneousArray import HeterogeneousArray
from tffmodel.types.HeterogeneousSparseArray import HeterogeneousSparseArray

import pickle
import numpy as np
import sparse

def convertLayerListToNumpyArray(layer_list):
    # NOTE: defining the target dtype like this is a workaround for the problem of heterogeneous arrays
    #   If the sub-arrays are heterogeneous, we can store them into an array of type object and
    #       numpy keeps the individual types of the sub-arrays
    #   If the sub-arrays have the same dimensions, numpy applys the type to all sub-arrays. Hence,
    #       we cannot use the object type but have to use the type of the sub-arrays (assumed equal)
    target_dtype = object # needed for storing arrays with different sizes in a numpy array
    if(len(layer_list) <= 1 or all([len(layer) == len(layer_list[0]) for layer in layer_list])):
        target_dtype = layer_list[0].dtype # use actual data type in case of homogeneous sub-arrays
    return np.array(layer_list, dtype=target_dtype)

class HeterogeneousDenseArray(HeterogeneousArray):
    is_sparse = False

    def __init__(self, data_arrays):
        if(not isinstance(data_arrays, np.ndarray)):
            data_arrays = [da.todense() if isinstance(da, sparse.COO) else da
                for da in data_arrays]
            data_arrays = convertLayerListToNumpyArray(data_arrays)
        super().__init__(data_arrays)

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
        # TODO: find a more efficient way to serialize the arrays
        serialized_array = []
        target_dtype = self.getDType()
        if(target_dtype == object):
            raise RuntimeError("Dtype 'object' is not supported.")
        for layer_array in self._data:
            serialized_array.extend([layer_array.astype(target_dtype).tobytes(),
                np.int32(layer_array.shape).tobytes(), target_dtype.str.encode("UTF-8")])
        serialized_array.append(pickle.dumps(self.getCompressionProperties()))
        return serialized_array

    @classmethod
    def deserialize(self_class, serialized_array):
        data_arrays = []
        for idx in range(0, len(serialized_array)-1, 3):
            layer_shape = np.frombuffer(serialized_array[idx+1], dtype=np.int32)
            layer_dtype = np.dtype(serialized_array[idx+2].decode("UTF-8"))
            layer_array = np.frombuffer(serialized_array[idx], dtype=layer_dtype)
            data_arrays.append(layer_array.reshape(layer_shape))
        deserialized_object = self_class(data_arrays)
        compression_properties = pickle.loads(serialized_array[len(serialized_array)-1])
        deserialized_object.setCompressionProperties(compression_properties)
        return deserialized_object

    def setCompressionProperties(self, compr_props):
        self.compression_properties = compr_props

    def setDType(self, dtype):
        self._data = convertLayerListToNumpyArray([darr.astype(dtype) for darr in self._data])

    def sparsify(self, mask):
        masked_data = [layer * m.reshape(layer.shape) for layer, m in zip(self._data, mask)]
        return HeterogeneousSparseArray(masked_data)

    def min(self):
        return np.min([np.min(darr) for darr in self._data])
    def max(self):
        return np.max([np.max(darr) for darr in self._data])
    def floor(self):
        for darr_idx, darr in enumerate(self._data):
            self.setLayer(darr_idx, np.floor(darr, dtype=self.getDType()))

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
