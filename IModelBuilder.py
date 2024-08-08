from abc import ABC, abstractmethod
import tensorflow as tf

class IModelBuilder(ABC):
    def __init__(self, config):
        self.config = config

    def buildModel(self, data):
        return self.buildModelElementSpec(data.element_spec)

    def buildModelElementSpec(self, data_element_spec):
        # construct a sequential model
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=data_element_spec[0].shape[1:]))
        self.buildKerasModelLayers(model)
        return model

    @abstractmethod
    def buildKerasModelLayers(self, keras_model):
        pass

    @abstractmethod
    def getLoss(self):
        pass

    @abstractmethod
    def getMetrics(self):
        pass

    def getLearningRate(self):
        return self.learning_rate

    @abstractmethod
    def getOptimizer(self):
        pass

    def getFedLearningRates(self):
        return self.server_learning_rate, self.client_learning_rate

    @abstractmethod
    def getFedApiOptimizers(self):
        pass

    @abstractmethod
    def getFedCoreOptimizers(self):
        pass
