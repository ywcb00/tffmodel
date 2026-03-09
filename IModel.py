from abc import ABC, abstractmethod

class IModel(ABC):
    def __init__(self, config):
        self.config = config

    @classmethod
    @abstractmethod
    def fromExistingModel(self_class, model, optimizer, config):
        pass

    @classmethod
    @abstractmethod
    def createModelElementSpec(self_class, data_element_spec, config):
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def getModel(self):
        pass

    @abstractmethod
    def getWeights(self):
        pass

    @abstractmethod
    def setWeights(self, weights):
        pass

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def fitGradient(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass
