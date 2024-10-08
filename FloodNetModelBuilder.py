from tffmodel.IModelBuilder import IModelBuilder

import tensorflow as tf
import tensorflow_federated as tff

class FloodNetModelBuilder(IModelBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = float(config.setdefault("lr", 0.00005))
        self.server_learning_rate = float(config.setdefault("lr_server", 0.2))
        self.client_learning_rate = float(config.setdefault("lr_client", 0.00005))
        self.model_abbrv = "c96_c32_dr25"

    def buildKerasModelLayers(self, keras_model):
        # NOTE: set the initializers in order to ensure reproducibility
        match self.model_abbrv:
            case "c10_avg_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(10, 10, strides=5, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c20_c10_avg_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(20, 10, strides=10, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.Conv2D(10, 8, strides=10, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c40_c20_c10_avg_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(40, 10, strides=5, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.Conv2D(20, 10, strides=10, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.Conv2D(10, 8, strides=10, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c32_c64_c16_avg_fl_dr50":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(32, (11, 11), strides=2, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(64, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(16, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.BatchNormalization(
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones(),
                    moving_mean_initializer=tf.keras.initializers.Zeros(),
                    moving_variance_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Flatten())
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c64_c32_avg_dr50_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c64_c32_c32_avg_dr50_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(64, (11, 11), strides=7, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(32, (7, 7), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c96_c64_c32_avg_dr50_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(64, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Conv2D(32, (7, 7), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.GlobalAveragePooling2D())
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c96_c32_dr25_dr50":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(96, (17, 17), strides=5, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=16,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((5, 5)))
                keras_model.add(tf.keras.layers.Conv2D(32, (11, 11), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.GroupNormalization(groups=8,
                    beta_initializer=tf.keras.initializers.Zeros(),
                    gamma_initializer=tf.keras.initializers.Ones()))
                keras_model.add(tf.keras.layers.Activation("relu"))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Flatten())
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dropout(0.5, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
            case "c96_c32_dr25":
                # first layer is the input
                keras_model.add(tf.keras.layers.Conv2D(96, (17, 17), strides=5, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    activation=tf.keras.activations.relu))
                keras_model.add(tf.keras.layers.MaxPool2D((5, 5)))
                keras_model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=3, padding="same",
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    activation=tf.keras.activations.relu))
                keras_model.add(tf.keras.layers.MaxPool2D((3, 3)))
                keras_model.add(tf.keras.layers.Dropout(0.25, seed=self.config["seed"]))
                keras_model.add(tf.keras.layers.Flatten())
                keras_model.add(tf.keras.layers.Dense(64, activation=tf.keras.activations.relu,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))
                keras_model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.config["seed"]),
                    bias_initializer=tf.keras.initializers.Zeros()))

    def getLoss(self):
        return tf.keras.losses.BinaryCrossentropy()

    def getMetrics(self):
        return [tf.keras.metrics.BinaryCrossentropy(),
            tf.keras.metrics.BinaryAccuracy()]

    def getLearningRateSchedule(self):
        lr = self.getLearningRate()
        return lr

    def getOptimizer(self):
        return tf.keras.optimizers.SGD(
            learning_rate=self.getLearningRateSchedule())

    def getFedLearningRateSchedules(self):
        server_lr, client_lr = self.getFedLearningRates()
        return server_lr, client_lr

    def getFedApiOptimizers(self):
        server_lr_sched, client_lr_sched = self.getFedLearningRateSchedules()
        server_optimizer = tf.keras.optimizers.SGD(learning_rate=server_lr_sched)
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=client_lr_sched)
        return server_optimizer, client_optimizer

    def getFedCoreOptimizers(self):
        # NOTE: only float learning rates in tff, no schedules
        server_lr, client_lr = self.getFedLearningRates()
        server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=server_lr)
        client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=client_lr)
        return server_optimizer, client_optimizer
