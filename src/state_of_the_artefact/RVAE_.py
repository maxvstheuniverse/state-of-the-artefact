import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def repeat_vector(args):
    """reconstructed_seq = Lambda(repeat_vector, output_shape=(None, size_of_vector))([vector, seq])"""
    layer_to_repeat, sequence_layer = args[0], args[1]
    return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)


def sampling(args):
    z_mean, z_logvar = args
    epsilon = tf.random.normal(tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * epsilon


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
        self.kl_loss = tf.keras.metrics.Mean(name="kl_loss")

        # -- Encoder
        inputs = keras.layers.Input(shape=(timesteps, original_dim,))

        h_encoder = RNN(hidden_dim, return_sequences=True)(inputs)

        z_mean = RNN(latent_dim)(h_encoder)
        z_logvar = RNN(latent_dim)(h_encoder)
        z = keras.layers.Lambda(sampling)([z_mean, z_logvar])

        self.encoder = keras.Model(inputs=[inputs], outputs=[z_mean, z_logvar, z], name="Encoder")

        # -- Decoder
        encoded = keras.layers.Input(shape=(latent_dim,))

        h_decoder = keras.layers.RepeatVector(timesteps)(encoded)
        h_decoder = RNN(hidden_dim, return_sequences=True)(h_decoder)

        outputs = keras.layers.TimeDistributed(keras.layers.Dense(original_dim, activation="softmax"))(h_decoder)

        self.decoder = keras.Model(inputs=[encoded], outputs=[outputs], name="Decoder")

    def call(self, x):
        z_mean, z_logvar, z = self.encoder(x)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
        self.add_loss(kl_loss)
        self.kl_loss.update_state(kl_loss)
        return reconstructed

    def encode(self, x):
        """ Encodes multiple samples. """
        return self.encoder(x)

    def decode(self, z):
        """ Decodes multiple latent variables. """
        return self.decoder(z)