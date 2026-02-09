from tffmodel.IModelBuilder import IModelBuilder

import tensorflow as tf

class IrisModelBuilder(IModelBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = float(config.setdefault("lr", 0.1))
        self.server_learning_rate = float(config.setdefault("lr_global", 1.))

    def buildKerasModelLayers(self, keras_model):
        # Adopted from https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        num_classes = 3

        # NOTE: set the initializers in order to ensure reproducibility
        keras_model.add(tf.keras.layers.Dense(8, activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
            bias_initializer=tf.keras.initializers.Zeros()))
        keras_model.add(tf.keras.layers.Dense(3, activation='softmax',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
            bias_initializer=tf.keras.initializers.Zeros()))

    def getLoss(self):
        return tf.keras.losses.CategoricalCrossentropy()

    def getMetrics(self):
        return [tf.metrics.CategoricalCrossentropy(),
            tf.metrics.CategoricalAccuracy()]

    def getLearningRateSchedule(self):
        lr = self.getLearningRate()
        return lr

    def getOptimizer(self):
        return tf.keras.optimizers.SGD(
            learning_rate=self.getLearningRateSchedule())

    def getFedLearningRateSchedules(self):
        server_lr, client_lr = self.getFedLearningRates()
        # client_lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=client_lr,
        #     decay_steps=10,
        #     decay_rate=0.6)
        return server_lr, client_lr

    def getFedApiOptimizers(self):
        server_lr_sched, client_lr_sched = self.getFedLearningRateSchedules()
        server_optimizer = tf.keras.optimizers.SGD(learning_rate=server_lr_sched)
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=client_lr_sched)
        return server_optimizer, client_optimizer

    def getFedCoreOptimizers(self):
        import tensorflow_federated as tff
        # NOTE: only float learning rates in tff, no schedules
        server_lr, client_lr = self.getFedLearningRates()
        server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=server_lr)
        client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=client_lr)
        return server_optimizer, client_optimizer
