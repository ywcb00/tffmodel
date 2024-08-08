from tffmodel.Gradient import Gradient
from tffmodel.IModel import IModel
from tffmodel.ModelBuilderUtils import getModelBuilder, getLoss, getMetrics, getOptimizer
from tffmodel.Weights import Weights

import logging
import numpy as np
import tensorflow as tf

class KerasModel(IModel):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("model/KerasModel")
        self.logger.setLevel(config["log_level"])

    @classmethod
    def createKerasModel(self_class, data, config):
        return self_class.createKerasModelElementSpec(data.element_spec, config)

    @classmethod
    def createKerasModelElementSpec(self_class, data_element_spec, config):
        model_builder = getModelBuilder(config)
        keras_model = model_builder.buildModelElementSpec(data_element_spec)
        return keras_model

    @classmethod
    def fromExistingModel(self_class, model, optimizer, config):
        keras_model = KerasModel(config)
        keras_model.model = model
        keras_model.initOptimizer(optimizer)
        return keras_model

    def clone(self):
        cloned_model = tf.keras.models.clone_model(self.model)
        duplicated_keras_model = KerasModel.fromExistingModel(
            cloned_model, self.model.optimizer, self.config)
        return duplicated_keras_model

    def getModel(self):
        return self.model

    def setWeights(self, weights):
        # assign the weights to the keras model
        self.model.set_weights(weights.get())

    def getWeights(self):
        return Weights(self.model.get_weights())

    def initModel(self, data):
        self.initModelWithOptimizer(data,
            optimizer=getOptimizer(self.config))

    def initModelWithOptimizer(self, data, optimizer):
        self.model = self.createKerasModel(data, self.config)
        self.initOptimizer(optimizer)

    def initOptimizer(self, optimizer):
        self.model.compile(optimizer=optimizer,
            loss=getLoss(self.config),
            metrics=getMetrics(self.config))

        if(self.config.setdefault('tensorboard_logging', True)):
            # set logging for tensorboard visualization
            logdir = f'{self.config["log_dir"]}/tensorboard' # delete any previous results
            try:
                tf.io.gfile.rmtree(logdir)
            except tf.errors.NotFoundError as e:
                pass # ignore if no previous results to delete
            self.logging_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            print(f'Log directory used for tensorboard: {logdir}')

    def fit(self, dataset):
        self.logger.info(f'Fitting local model with {dataset.train.cardinality()} train instances.')

        fit_history = self.model.fit(x=dataset.train,
            y=None, # already in the dataset
            batch_size=None, # already in the dataset
            epochs=self.config["num_train_rounds"],
            validation_data=None, # we have a separate validation split
            shuffle=False,
            verbose=2,
            callbacks=None if not self.config.setdefault('tensorboard_logging', True) else [self.logging_callback])

        return fit_history

    # def fitGradients(self, dataset):
    #     self.logger.info(f'Fitting local model with {dataset.train.cardinality()} train instances ' +
    #         'using explicit gradients.')

    #     for epoch in range(self.config["num_train_rounds"]):
    #         for step, (x_batch_train, y_batch_train) in enumerate(dataset.train):
    #             with tf.GradientTape() as tape:
    #                 preds = self.model(x_batch_train, training=True)
    #                 loss_value = getLoss(self.config)(y_batch_train, preds)
    #             grads = tape.gradient(loss_value, self.model.trainable_variables)
    #             self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #             if(step == 0):
    #                 accumulated_grads = np.array(grads, dtype=object)
    #             else:
    #                 accumulated_grads += np.array(grads, dtype=object)

    #     return accumulated_grads

    # compute the accumulated gradient, no training
    def computeGradient(self, dataset):
        for epoch in range(self.config["num_train_rounds"]):
            for step, (x_batch_train, y_batch_train) in enumerate(dataset.train):
                with tf.GradientTape() as tape:
                    preds = self.model(x_batch_train, training=True)
                    loss_value = getLoss(self.config)(y_batch_train, preds)
                grad = tape.gradient(loss_value, self.model.trainable_variables)
                grad = Gradient([g.numpy() for g in grad])
                if(step == 0):
                    accumulated_grad = grad
                else:
                    accumulated_grad += grad

        return accumulated_grad

    def predict(self, data):
        return self.predictKerasModel(self.model, data)

    @classmethod
    def predictKerasModel(self_class, keras_model, data):
        predictions = keras_model.predict(data)
        return predictions

    def evaluate(self, data):
        evaluation_metrics = self.evaluateKerasModel(self.model, data)
        self.logger.info(f'Evaluation on {data.cardinality()} instances resulted in {evaluation_metrics}')
        return evaluation_metrics

    @classmethod
    def evaluateKerasModel(self_class, keras_model, data):
        evaluation_scalars = keras_model.evaluate(data, verbose=2)
        evaluation_metrics = dict(zip(keras_model.metrics_names, evaluation_scalars))
        return evaluation_metrics
