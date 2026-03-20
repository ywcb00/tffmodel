from tffmodel.IPyTorchModelBuilder import IPyTorchModelBuilder

import torch

class IrisPyTorchModelBuilder(IPyTorchModelBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = float(config.setdefault("lr", 0.1))
        self.server_learning_rate = float(config.setdefault("lr_global", 1.))

    def buildModelElementSpec(self, data_element_spec):
        # Adopted from https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        class IrisModel(torch.nn.Module):
            def __init__(self):
                super(IrisModel, self).__init__()
                self.layer1 = torch.nn.Linear(4, 8)
                self.activation1 = torch.nn.ReLU()
                self.layer2 = torch.nn.Linear(8, 3)
                self.activation2 = torch.nn.Softmax()
                self.double()
            
            def forward(self, x):
                x = self.layer1(x)
                x = self.activation1(x)
                x = self.layer2(x)
                x = self.activation2(x)
                return x

        model = IrisModel()
        return model

    def getLoss(self):
        def oneHotCrossEntropyLoss(y_pred, y_true):
            y_pred = y_pred.to(torch.float32)
            y_true = y_true.to(torch.float32)
            loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
            return loss
        return oneHotCrossEntropyLoss

    def getMetrics(self):
        def oneHotCrossEntropyLoss(y_pred, y_true):
            y_pred = y_pred.to(torch.float32)
            y_true = y_true.to(torch.float32)
            loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
            return loss
        metrics = {
            "CCE": oneHotCrossEntropyLoss
        }
        return metrics

    def getLearningRateSchedule(self):
        raise NotImplementedError

    def getOptimizer(self):
        return torch.optim.SGD

    def getFedLearningRateSchedules(self):
        raise NotImplementedError

    def getFedApiOptimizers(self):
        raise NotImplementedError

    def getFedCoreOptimizers(self):
        raise NotImplementedError
