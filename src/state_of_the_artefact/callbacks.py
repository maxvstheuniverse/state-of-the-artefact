import tensorflow as tf


class KLAnnealing(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch, end_epoch=None):
        super(KLAnnealing, self).__init__()

        if end_epoch is None:
            self.start = 0
            self.end = start_epoch
        else:
            self.start = start_epoch
            self.end = end_epoch

    def on_epoch_begin(self, epoch, logs={}):
        if not hasattr(self.model, "kl_weight"):
            raise ValueError('Model must have a "kl_weight" attribute.')

        if epoch <= self.start:
            tf.keras.backend.set_value(self.model.kl_weight, 0.0)
            print("KL Annealing weight set to 0.0")

    def on_epoch_end(self, epoch, logs={}):
        if not hasattr(self.model, "kl_weight"):
            raise ValueError('Model must have a "kl_weight" attribute.')

        if epoch > self.start and epoch < self.end:
            x = (epoch - self.start) / (self.end - self.start)
            kl_weight = 1 / (1 + tf.exp(-10 * x + 5))

            tf.keras.backend.set_value(self.model.kl_weight, kl_weight)
            print(f"KL Annealing weight set to {kl_weight}")

        elif epoch == self.end:
            tf.keras.backend.set_value(self.model.kl_weight, 1.0)
            print("KL Annealing weight set to 1.0")
