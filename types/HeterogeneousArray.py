from abc import ABC, abstractclassmethod, abstractmethod
import numpy as np

class HeterogeneousArray(ABC):
    def __init__(self, data):
        self._data = data
        # stores the properties needed to decompress the data if compressed
        self.compression_properties = None

    def getLength(self):
        return len(self._data)

    def getCompressionProperties(self):
        return self.compression_properties

    @abstractmethod
    def setCompressionProperties(self, compr_props):
        pass

    @abstractclassmethod
    def getZero(self_class, shape_layer_arrays):
        pass

    @abstractmethod
    def get(self):
        pass

    def getNumLayers(self):
        return len(self._data)

    def getShapes(self):
        shapes = [arr.shape for arr in self._data]
        return shapes

    def getSizes(self):
        sizes = [arr.size for arr in self._data]
        return sizes

    def getSize(self):
        size = sum(self.getSizes())
        return size

    def getDType(self):
        if(self.getNumLayers() == 0):
            return np.dtype("void")
        dtypes = [arr.dtype for arr in self._data]
        assert all(dtp == dtypes[0] for dtp in dtypes), "Different dtypes are not supported yet"
        return dtypes[0]

    @abstractmethod
    def setDType(self, dtype):
        pass

    def getDTypeName(self):
        return self.getDType().name

    def take(self, indices):
        data_arrays = [self._data[idx] for idx in indices]
        return self.__class__(data_arrays)

    def setLayer(self, layer_idx, layer_array):
        self._data[layer_idx] = layer_array

    @abstractmethod
    def serialize(self):
        pass
    @abstractclassmethod
    def deserialize(self_class, serialized_array):
        pass

    @abstractmethod
    def sparsify(self, mask):
        pass

    @abstractmethod
    def min(self):
        pass
    @abstractmethod
    def max(self):
        pass
    @abstractmethod
    def floor(self):
        pass

    def __repr__(self):
        return (f'{self.__class__.__name__}(#arrays={self.getLength()}, '
            f'shape={self.getShapes()}, dtype={self.getDType()})')

    @abstractclassmethod
    def add_primitive(self_class, lhs, rhs):
        pass
    @abstractclassmethod
    def add_operation(self_class, lhs, rhs):
        pass
    @abstractclassmethod
    def sub_primitive(self_class, lhs, rhs):
        pass
    @abstractclassmethod
    def sub_operation(self_class, lhs, rhs):
        pass
    @abstractclassmethod
    def mul_primitive(self_class, lhs, rhs):
        pass
    @abstractclassmethod
    def mul_operation(self_class, lhs, rhs):
        pass
    @abstractclassmethod
    def div_primitive(self_class, lhs, rhs):
        pass
    @abstractclassmethod
    def div_operation(self_class, lhs, rhs):
        pass
    @abstractclassmethod
    def eq_primitive(self_class, lhs, rhs):
        pass

    # Operator overload
    def __add__(self, other): # +
        res = self.add_operation(self, other)
        return res
    def __sub__(self, other): # -
        res = self.sub_operation(self, other)
        return res
    def __mul__(self, other): # *
        res = self.mul_operation(self, other)
        return res
    def __truediv__(self, other): # /
        res = self.div_operation(self, other)
        return res
    def __eq__(self, other): # ==
        res = self.eq_primitive(self, other)
        return res
    def __ne__(self, other): # !=
        return not self.__eq__(other)
    def __iadd__(self, other): # +=
        self._data = self.add_primitive(self, other)
        return self
    def __isub__(self, other): # -=
        self._data = self.sub_primitive(self, other)
        return self
    def __imul__(self, other): # *=
        self._data = self.mul_primitive(self, other)
        return self
    def __idiv__(self, other): # /=
        self._data = self.div_primitive(self, other)
        return self
    def __getitem__(self, key): # []
        return self._data[key]
