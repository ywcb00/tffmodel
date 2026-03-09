from abc import ABC, abstractmethod

class IModelBuilder(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def buildModelElementSpec(self, data_element_spec):
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
    def getLearningRateSchedule(self):
        pass

    @abstractmethod
    def getOptimizer(self):
        pass

    def getFedLearningRates(self):
        return self.server_learning_rate, self.learning_rate

    @abstractmethod
    def getFedLearningRateSchedules(self):
        pass

    @abstractmethod
    def getFedApiOptimizers(self):
        pass

    @abstractmethod
    def getFedCoreOptimizers(self):
        pass
