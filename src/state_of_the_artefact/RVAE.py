import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def repeat_vector(args):
    """reconstructed_seq = Lambda(repeat_vector, output_shape=(None, size_of_vector))([vector, seq])"""
    layer_to_repeat, sequence_layer = args[0], args[1]
    return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


def sampling(args):
    z_mean, z_log_sigma = args
    batch, dim = K.shape(z_mean)[0], K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_sigma) * epsilon


class RecurrentVariationalAutoEncoder(tf.keras.Model):
    def __init__(self,
                 timesteps,
                 original_dim,
                 hidden_dim,
                 latent_dim,
                 RNN=keras.layers.LSTM,
                 name="RVAE",
                 **kwargs):
        super(RecurrentVariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim

        # -- Encoder
        inputs = keras.layers.Input(shape=(timesteps, original_dim,))

        # z = RNN(100, return_sequences=True)(inputs)
        z = RNN(hidden_dim)(inputs)

        # -- encoder outputs
        z_mean = keras.layers.Dense(latent_dim)(z)
        z_logvar = keras.layers.Dense(latent_dim)(z)
        z = keras.layers.Lambda(sampling)([z_mean, z_logvar])

        self.encoder = keras.Model(inputs=[inputs], outputs=[z_mean, z_logvar, z], name="Encoder")

        # -- Decoder
        encoded = keras.layers.Input(shape=(latent_dim,))

        # x = keras.layers.Lambda(repeat_vector, output_shape=(None, original_dim))([encoded, inputs])
        x = keras.layers.RepeatVector(timesteps)(encoded)

        # x = RNN(30, return_sequences=True)(x)
        x = RNN(hidden_dim, return_sequences=True)(x)

        # -- decoder outputs
        x = keras.layers.TimeDistributed(keras.layers.Dense(original_dim, activation="softmax"))(x)

        self.decoder = keras.Model(inputs=[encoded], outputs=[x], name="Decoder")

    def call(self, x):
        z_mean, z_logvar, z = self.encoder(x)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
        self.add_loss(kl_loss)
        return reconstructed

    def encode(self, x):
        """ Encodes multiple samples. """
        return self.encoder(x)

    def decode(self, z):
        """ Decodes multiple latent variables. """
        return self.decoder(z)
