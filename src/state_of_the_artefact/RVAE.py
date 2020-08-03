import tensorflow as tf
from tensorflow.keras import Model, layers, metrics
from state_of_the_artefact.utilities import make_onehot


def sampling(args):
    z_mean, z_logvar = args
    epsilon = tf.random.normal(tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * epsilon


def reconstruction_loss(x_logits, x):
    return tf.nn.softmax_cross_entropy_with_logits(logits=x_logits, labels=x)


def kl_loss(z_mean, z_logvar):
    return -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))


class RecurrentVariationalAutoEncoder(tf.keras.Model):
    def __init__(self,
                 timesteps,
                 original_dim,
                 hidden_dim,
                 latent_dim,
                 RNN=tf.keras.layers.LSTM,
                 name="RVAE",
                 beta=1.0,
                 **kwargs):
        super(RecurrentVariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.timesteps = timesteps
        self.original_dim = original_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.vae_loss = metrics.Mean(name="vae_loss")
        self.reconstruction_loss = metrics.Mean(name="reconstruction_loss")
        self.kl_loss = metrics.Mean(name="kl_loss")
        self.kl_annealed = metrics.Mean(name="kl_annealed")

        self.kl_weight = tf.Variable(beta, trainable=False)

        # -- Encoder
        inputs = layers.Input(shape=(timesteps, original_dim,))

        h_encoder = layers.LSTM(hidden_dim, return_sequences=True)(inputs)
        h_encoder = layers.LSTM(hidden_dim)(h_encoder)

        z_mean = layers.Dense(latent_dim, activation="linear")(h_encoder)
        z_logvar = layers.Dense(latent_dim, activation="linear")(h_encoder)
        z = layers.Lambda(sampling)([z_mean, z_logvar])

        self.encoder = Model(inputs=[inputs], outputs=[z_mean, z_logvar, z], name="Encoder")

        # -- Decoder
        z_inputs = layers.Input(shape=(latent_dim,))

        h_decoder = layers.RepeatVector(timesteps)(z_inputs)
        h_decoder = layers.LSTM(hidden_dim, return_sequences=True)(h_decoder)
        h_decoder = layers.LSTM(hidden_dim, return_sequences=True)(h_decoder)

        outputs = layers.TimeDistributed(layers.Dense(original_dim))(h_decoder)

        self.decoder = Model(inputs=[z_inputs], outputs=[outputs], name="Decoder")

    @tf.function
    def train_step(self, x, y=None):
        if isinstance(x, tuple):
            y = x[1]
            x = x[0]

        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encode(x)
            x_logits = self.decode(z)

            CE = reconstruction_loss(x_logits, y)
            KL = kl_loss(z_mean, z_logvar)

            vae_loss = CE + self.kl_weight * KL

        gradients = tape.gradient(vae_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.reconstruction_loss.update_state(CE)
        self.kl_loss.update_state(KL)
        self.kl_annealed.update_state(self.kl_weight * KL)
        self.vae_loss.update_state(vae_loss)

        return {"vae_loss": self.vae_loss.result(),
                "reconstruction_loss": self.reconstruction_loss.result(),
                "kl_loss": self.kl_loss.result(),
                "kl_annealed": self.kl_annealed.result()}

    @tf.function
    def test_step(self, x, y=None):
        if isinstance(x, tuple):
            y = x[1]
            x = x[0]

        z_mean, z_logvar, z = self.encode(x)
        x_logit = self.decode(z)

        CE = reconstruction_loss(x_logit, y)
        KL = kl_loss(z_mean, z_logvar)
        vae_loss = CE + KL

        self.reconstruction_loss.update_state(CE)
        self.kl_loss.update_state(KL)
        self.kl_annealed.update_state(self.kl_weight * KL)
        self.vae_loss.update_state(vae_loss)

        return {"vae_loss": self.vae_loss.result(),
                "reconstruction_loss": self.reconstruction_loss.result(),
                "kl_loss": self.kl_loss.result(),
                "kl_annealed": self.kl_annealed.result()}

    @tf.function
    def sample(self, epsilon=None, size=1):
        """ Generates a random sample. """
        if epsilon is None:
            epsilon = tf.random.normal(shape=(size, self.latent_dim))
        return self.decode(epsilon, apply_softmax=True)

    def encode(self, x):
        """ Encodes multiple samples. """
        z_mean, z_logvar, z = self.encoder(x)
        return z_mean, z_logvar, z

    def decode(self, z, apply_softmax=False, apply_onehot=False):
        """ Decodes multiple latent variables.

            If `apply_onehot` is `True` the function will also apply the softmax.
        """
        x_hat = self.decoder(z)

        if apply_softmax or apply_onehot:
            x_hat = tf.nn.softmax(x_hat, axis=-1)

            if apply_onehot:
                x_hat = make_onehot(x_hat)

        return x_hat

    def call(self, x):
        _, _, z = self.encode(x)
        return self.decode(z, apply_softmax=True)
