from abc import abstractmethod

from tffmodel.IModelBuilder import IModelBuilder

class IPyTorchModelBuilder(IModelBuilder):
    def __init__(self, config):
        super().__init__(config)
