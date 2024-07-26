from tffmodel.IModel import IModel
from tffmodel.ModelBuilderUtils import getModelBuilder, getLoss, getMetrics, getOptimizer

import logging
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

    def setWeights(self, weights):
        # assign the weights to the keras model
        self.model.set_weights(weights)

    def getWeights(self):
        return self.model.get_weights()

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

        # set logging for tensorboard visualization
        logdir = self.config["log_dir"] # delete any previous results
        try:
            tf.io.gfile.rmtree(logdir)
        except tf.errors.NotFoundError as e:
            pass # ignore if no previous results to delete
        self.logging_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    def fit(self, dataset):
        self.logger.info(f'Fitting local model with {dataset.train.cardinality()} train instances')

        self.fit_history = self.model.fit(x=dataset.train,
            y=None, # already in the dataset
            batch_size=None, # already in the dataset
            epochs=self.config["num_train_rounds"],
            validation_data=None, # we have a separate validation split
            shuffle=False,
            verbose=2,
            callbacks=[self.logging_callback])

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
