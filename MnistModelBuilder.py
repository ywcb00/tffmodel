from tffmodel.IModelBuilder import IModelBuilder

import tensorflow as tf
import tensorflow_federated as tff

class MnistModelBuilder(IModelBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = float(config.setdefault("lr", 0.001))
        self.server_learning_rate = float(config.setdefault("lr_server", 1.))
        self.client_learning_rate = float(config.setdefault("lr_client", 0.02))

    def buildKerasModelLayers(self, keras_model):
        num_classes = 10

        # NOTE: set the initializers in order to ensure reproducibility
        keras_model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
            bias_initializer=tf.keras.initializers.Zeros()))
        keras_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        keras_model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
            bias_initializer=tf.keras.initializers.Zeros()))
        keras_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        keras_model.add(tf.keras.layers.Flatten())
        keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
        keras_model.add(tf.keras.layers.Dense(num_classes, activation="softmax",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
            bias_initializer=tf.keras.initializers.Zeros()))

    def getLoss(self):
        return tf.keras.losses.CategoricalCrossentropy()

    def getMetrics(self):
        return [tf.metrics.CategoricalCrossentropy(),
            tf.metrics.CategoricalAccuracy()]

    def getOptimizer(self):
        return tf.keras.optimizers.SGD(learning_rate=self.getLearningRate())

    def getFedApiOptimizers(self):
        server_lr, client_lr = self.getFedLearningRates()
        server_optimizer = tf.keras.optimizers.SGD(learning_rate=server_lr)
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=client_lr)
        return server_optimizer, client_optimizer

    def getFedCoreOptimizers(self):
        server_lr, client_lr = self.getFedLearningRates()
        server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=server_lr)
        client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=client_lr)
        return server_optimizer, client_optimizer
