from abc import ABC, abstractclassmethod, abstractmethod
import numpy as np

class HeterogeneousArray(ABC):
    def __init__(self, data_arrays):
        self._data = np.array(data_arrays, dtype=object)

    @abstractclassmethod
    def getZero(self_class, shape_layer_arrays):
        pass

    def get(self):
        return self._data

    def __repr__(self):
        return (f'{self.__class__.__name__}(#arrays={len(self._data)}, '
            f'shape={[da.shape for da in self._data]}, dtype={[da.dtype for da in self._data]})')

    @abstractmethod
    def serialize(self):
        pass

    # param shape_arrays is used to identify the respective shapes
    @abstractclassmethod
    def deserialize(self_class, serialized_array, shape_arrays):
        pass

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
