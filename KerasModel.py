from tffmodel.IModel import IModel
from tffmodel.ModelBuilderUtils import getModelBuilder, getLoss, getMetrics, getOptimizer
from tffmodel.types.HeterogeneousDenseArray import HeterogeneousDenseArray

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
        restore_weights = self.getWeights()
        cloned_model = tf.keras.models.clone_model(self.model)
        # TODO: FIXME: workaround for cloning an optimizer by serializing and deserializing
        cloned_optimizer = tf.keras.optimizers.deserialize(
            tf.keras.optimizers.serialize(self.model.optimizer))
        cloned_keras_model = KerasModel.fromExistingModel(
            cloned_model, cloned_optimizer, self.config)
        cloned_keras_model.setWeights(restore_weights)
        return cloned_keras_model

    def getModel(self):
        return self.model

    def setWeights(self, weights):
        # assign the weights to the keras model
        self.model.set_weights(weights.get())

    def getWeights(self):
        return HeterogeneousDenseArray(self.model.get_weights())

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

        if(self.config.setdefault('log_tensorboard_flag', True)):
            # set logging for tensorboard visualization
            logdir = f'{self.config["log_dir"]}/tensorboard' # delete any previous results
            try:
                tf.io.gfile.rmtree(logdir)
            except tf.errors.NotFoundError as e:
                pass # ignore if no previous results to delete
            self.logging_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            print(f'Log directory used for tensorboard: {logdir}')

    def addKernelRegularizers(self, regs):
        counter = 0
        for layer in self.model.layers:
            if(hasattr(layer, 'kernel_regularizer')):
                setattr(layer, 'kernel_regularizer', regs[counter])
                counter += 1
        restore_weights = self.model.get_weights()
        # TODO: FIXME: workaround for optimizer cloning through serialization and deserialization
        cloned_optimizer = tf.keras.optimizers.deserialize(
            tf.keras.optimizers.serialize(self.model.optimizer))
        # TODO: FIXME: workaround: kernel regularizers are changed only in the config --> clone the model from config
        self.model = tf.keras.models.clone_model(self.model)
        # compile model
        self.initOptimizer(cloned_optimizer)
        self.model.set_weights(restore_weights)

    def fit(self, dataset):
        self.logger.info(f'Fitting local model for {self.config["num_local_epochs"]} local epochs ' +
            f'with {dataset.train.cardinality()} train instances.')

        fit_history = self.model.fit(x=dataset.train,
            y=None, # already in the dataset
            batch_size=None, # already in the dataset
            epochs=self.config["num_local_epochs"],
            validation_data=None, # we have a separate validation split
            shuffle=False,
            verbose=2,
            callbacks=None if not self.config.setdefault('log_tensorboard_flag', True) else [self.logging_callback])

        return fit_history

    def fitGradient(self, dataset):
        self.logger.info(f'Fitting model for {self.config["num_local_epochs"]} local epochs ' +
            f'with {dataset.train.cardinality()} train instances using explicit gradients.')

        train_metrics = None

        for epoch in range(self.config["num_local_epochs"]):
            for step, (x_batch_train, y_batch_train) in enumerate(dataset.train):
                with tf.GradientTape() as tape:
                    preds = self.model(x_batch_train, training=True)
                    loss_value = getLoss(self.config)(y_batch_train, preds)
                grad = tape.gradient(loss_value, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
                grad = HeterogeneousDenseArray([g.numpy() for g in grad])
                if(step == 0):
                    accumulated_grad = grad
                else:
                    accumulated_grad += grad

            if(self.config.setdefault('log_performance_flag', False)):
                scalar_train_metrics = KerasModel.evaluateKerasModel(
                    self.getModel(), dataset.train)
                if(train_metrics == None):
                    train_metrics = {mname: [mval] for mname, mval in scalar_train_metrics.items()}
                else:
                    for mname, mval in scalar_train_metrics.items():
                        train_metrics[mname].append(mval)

        return accumulated_grad, train_metrics

    # compute the accumulated gradient, no training
    # "The sum of the gradients is the same as the gradient obtained on the full batch"
    def computeGradient(self, dataset, num_local_epochs=None):
        if(num_local_epochs == None):
            num_local_epochs = self.config["num_local_epochs"]
        if(num_local_epochs != 1):
            raise NotImplementedError("Multiple local epochs not supported yet.")
        for epoch in range(num_local_epochs):
            for step, (x_batch_train, y_batch_train) in enumerate(dataset.train):
                with tf.GradientTape() as tape:
                    preds = self.model(x_batch_train, training=True)
                    loss_value = getLoss(self.config)(y_batch_train, preds)
                grad = tape.gradient(loss_value, self.model.trainable_variables)
                grad = HeterogeneousDenseArray([g.numpy() for g in grad])
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
