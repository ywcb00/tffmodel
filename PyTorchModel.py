from tffdataset.DatasetUtils import getDatasetClass
from tffmodel.IModel import IModel
from tffmodel.ModelBuilderUtils import getModelBuilder, getLearningRate, getLoss, getMetrics, getOptimizer
from tffmodel.types.HeterogeneousDenseArray import HeterogeneousDenseArray

import copy
import io
import logging
import numpy as np
import pickle
import torch


class PyTorchModel(IModel):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("model/PyTorchModel")
        self.logger.setLevel(config["log_level"])

    @classmethod
    def createModel(self_class, data, config):
        return self_class.createModelElementSpec(data.element_spec, config)

    @classmethod
    def createModelElementSpec(self_class, data_element_spec, config):
        model_builder = getModelBuilder(config)
        model = model_builder.buildModelElementSpec(data_element_spec)
        return model

    @classmethod
    def fromExistingModel(self_class, model, optimizer, config):
        pytorch_model = PyTorchModel(config)
        pytorch_model.model = model
        pytorch_model.initOptimizer(optimizer)
        return pytorch_model

    # serialize the model architecture and the optimizer configuration of a pytorch model
    def serialize(self):
        example_inputs = getDatasetClass(self.config).getExampleInputs()
        exported_model = torch.export.export(self.model, (example_inputs,))
        buffer = io.BytesIO()
        torch.export.save(exported_model, buffer)
        serialized_model = buffer.getvalue()
        optimizer_config = {
            "class": self.optimizer.__class__.__name__
        }
        serialized_optimizer_config = pickle.dumps(optimizer_config, protocol=4)
        return serialized_model, serialized_optimizer_config

    # deserialize the model architecture and the optimizer configuration of a pytorch model
    @classmethod
    def deserializeModel(self_class, serialized_model_config, serialized_optimizer_config):
        exported_model = torch.export.load(io.BytesIO(serialized_model_config))
        deserialized_model = exported_model.module()
        optimizer_config = pickle.loads(serialized_optimizer_config)
        deserialized_optimizer = getattr(torch.optim, optimizer_config["class"])
        return deserialized_model, deserialized_optimizer

    def clone(self):
        # deepcopy also copies the weights
        cloned_model = copy.deepcopy(self.model)
        cloned_optimizer = type(self.optimizer)
        cloned_pytorch_model = self.fromExistingModel(cloned_model, cloned_optimizer, self.config)
        return cloned_pytorch_model

    def getModel(self):
        return self.model

    def setWeights(self, weights):
        for layer_key, weights_layer in zip(self.model.state_dict().keys(), weights):
            self.model.state_dict()[layer_key] = weights_layer

    def getWeights(self):
        np_weights = [layer_weights.numpy() for layer_weights in self.model.state_dict().values()]
        return HeterogeneousDenseArray(np_weights)

    def initModel(self, data):
        self.initModelWithOptimizer(data,
            optimizer=getOptimizer(self.config))

    def initModelWithOptimizer(self, data, optimizer):
        self.model = self.createModel(data, self.config)
        self.initOptimizer(optimizer)

    def initModelElementSpec(self, data_element_spec):
        self.model = self.createModelElementSpec(data_element_spec, self.config)
        self.initOptimizer(getOptimizer(self.config))

    def initOptimizer(self, optimizer):
        self.optimizer = optimizer(self.model.parameters(), lr=getLearningRate(self.config))

    def fit(self, dataset):
        self.logger.info(f'Fitting local model for {self.config["num_local_epochs"]} local epochs ' +
            f'with {len(dataset.train)} train instances.')

        # self.model.train() # set model to training mode

        for epoch in range(self.config["num_local_epochs"]):
            for batch_X, batch_y in dataset.train:
                batch_y_pred = self.model(batch_X)
                loss = getLoss(self.config)(batch_y_pred, batch_y)
                self.optimizer.zero_grad() # reset gradients
                loss.backward()
                self.optimizer.step()

        # self.model.eval() # set model to evaluation mode

        train_metrics = self.evaluatePyTorchModel(self.model, dataset.train, self.config)

        return train_metrics

    def fitGradient(self, dataset):
        self.logger.info(f'Fitting model for {self.config["num_local_epochs"]} local epochs ' +
            f'with {dataset.train.cardinality()} train instances using explicit gradients.')

        raise NotImplementedError

    # compute the accumulated gradient, no training
    # "The sum of the gradients is the same as the gradient obtained on the full batch"
    def computeGradient(self, dataset, num_local_epochs=None):
        raise NotImplementedError

    def predict(self, data):
        return self.predictPyTorchModel(self.model, data)

    @classmethod
    def predictPyTorchModel(self_class, model, data):
        raise NotImplementedError

    def evaluate(self, data):
        evaluation_metrics = self.evaluatePyTorchModel(self.model, data, self.config)
        self.logger.info(f'Evaluation on {len(data.dataset)} instances resulted in {evaluation_metrics}')
        return evaluation_metrics

    @classmethod
    def evaluatePyTorchModel(self_class, model, dataloader, config):
        y_pred_list = list()
        y_true_list = list()
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                y_pred_batch = model(batch_X)
                y_pred_list.append(y_pred_batch)
                y_true_list.append(batch_y)
        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        eval_metrics = {key: val(y_pred, y_true) for key, val in getMetrics(config).items()}
        return eval_metrics
