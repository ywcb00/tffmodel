from tffdataset.DatasetUtils import DatasetID
from tffmodel.FloodNetModelBuilder import FloodNetModelBuilder
from tffmodel.IrisModelBuilder import IrisModelBuilder
from tffmodel.MnistModelBuilder import MnistModelBuilder

def getModelBuilder(config):
    # TODO: this implementation creates a model builder object for every call
    match config["dataset_id"]:
        case DatasetID.FloodNet:
            return FloodNetModelBuilder(config)
        case DatasetID.Mnist:
            return MnistModelBuilder(config)
        case DatasetID.Iris:
            return IrisModelBuilder(config)
        case _:
            raise NotImplementedError

def getLoss(config):
    return getModelBuilder(config).getLoss()

def getMetrics(config):
    return getModelBuilder(config).getMetrics()

def getLearningRate(config):
    return getModelBuilder(config).getLearningRate()

def getLearningRateSchedule(config):
    return getModelBuilder(config).getLearningRateSchedule()

def getOptimizer(config):
    return getModelBuilder(config).getOptimizer()

def getFedLearningRates(config):
    return getModelBuilder(config).getFedLearningRates()

def getFedLearningRateSchedules(config):
    return getModelBuilder(config).getFedLearningRateSchedules()

def getFedOptimizers(config):
    return getFedApiOptimizers(config)

def getFedApiOptimizers(config):
    return getModelBuilder(config).getFedApiOptimizers()

def getFedCoreOptimizers(config):
    return getModelBuilder(config).getFedCoreOptimizers()
