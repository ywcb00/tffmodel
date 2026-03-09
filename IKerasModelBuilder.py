from abc import ABC, abstractmethod
import tensorflow as tf

from tffmodel.IModelBuilder import IModelBuilder

class IKerasModelBuilder(IModelBuilder):
    def __init__(self, config):
        super().__init__(config)

    def buildModelElementSpec(self, data_element_spec):
        # construct a sequential model
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=data_element_spec[0].shape[1:]))
        self.buildKerasModelLayers(model)
        return model

    @abstractmethod
    def buildKerasModelLayers(self, model):
        pass
