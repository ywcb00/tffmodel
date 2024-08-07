import numpy as np

class Weights():
    def __init__(self, layer_weights):
        self._weights = np.array(layer_weights, dtype=object)

    @classmethod
    def getZero(self_class, shape_layer_weights):
        layer_weights = [np.zeros(slw.shape) for slw in shape_layer_weights]
        return self_class(layer_weights)

    def get(self):
        return self._weights

    def __repr__(self):
        return (f'{self.__class__.__name__}(#layers={len(self._weights)}, '
            f'shape={[lw.shape for lw in self._weights]}, dtype={[lw.dtype for lw in self._weights]})')

    def add_primitive(self, other):
        if(isinstance(other, self.__class__)):
            res = np.add(self._weights, other._weights)
        else:
            res = np.add(self._weights, other)
        return res
    def sub_primitive(self, other):
        if(isinstance(other, self.__class__)):
            res = np.subtract(self._weights, other._weights)
        else:
            res = np.subtract(self._weights, other)
        return res
    def mul_primitive(self, other):
        if(isinstance(other, self.__class__)):
            res = np.multiply(self._weights, other._weights)
        else:
            res = np.multiply(self._weights, other)
        return res
    def div_primitive(self, other):
        if(isinstance(other, self.__class__)):
            res = np.divide(self._weights, other._weights)
        else:
            res = np.divide(self._weights, other)
        return res

    # Operator overload
    def __add__(self, other): # +
        res = self.add_primitive(other)
        return self.__class__(res)
    def __sub__(self, other): # -
        res = self.sub_primitive(other)
        return self.__class__(res)
    def __mul__(self, other): # *
        res = self.mul_primitive(other)
        return self.__class__(res)
    def __truediv__(self, other): # /
        res = self.div_primitive(other)
        return self.__class__(res)
    def __eq__(self, other): # ==
        if(isinstance(other, self.__class__)):
            res = np.all(np.vectorize(np.array_equal)(self._weights, other._weights))
            return res
        else:
            return NotImplemented
    def __ne__(self, other): # !=
        return not self.__eq__(other)
    def __iadd__(self, other): # +=
        self._weights = self.add_primitive(other)
        return self
    def __isub__(self, other): # -=
        self._weights = self.sub_primitive(other)
        return self
    def __imul__(self, other): # *=
        self._weights = self.mul_primitive(other)
        return self
    def __idiv__(self, other): # /=
        self._weights = self.div_primitive(other)
        return self
    def __getitem__(self, key): # []
        return self._weights[key]
