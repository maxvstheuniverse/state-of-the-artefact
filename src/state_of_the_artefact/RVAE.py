import tensorflow as tf


# def repeat_vector(args):
#     """reconstructed_seq = Lambda(repeat_vector, output_shape=(None, size_of_vector))([vector, seq])"""
#     layer_to_repeat, sequence_layer = args[0], args[1]
#     return RepeatVector(tf.shape(sequence_layer)[1])(layer_to_repeat)


def sampling(args):
    z_mean, z_logvar = args
    epsilon = tf.random.normal(tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_logvar) * epsilon


def reconstruction_loss(x_logit, x):
    # in this paper they also take the mean over each timestep.
    # https://arxiv.org/pdf/1412.6581.pdf
    return tf.nn.softmax_cross_entropy_with_logits(logits=x_logit, labels=x)


def kl_loss(z_mean, z_logvar):
    # THe paper below notes that this should be reduce sum isntead of mean.
    # https://arxiv.org/pdf/1412.6581.pdf
    return -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))


class RecurrentVariationalAutoEncoder(tf.keras.Model):
    def __init__(self,
                 timesteps,
                 original_dim,
                 hidden_dim,
                 latent_dim,
                 RNN=tf.keras.layers.LSTM,
                 name="RVAE",
                 **kwargs):
        super(RecurrentVariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim

        self.elbo = tf.keras.metrics.Mean(name="elbo")
        self.kl_loss = tf.keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss = tf.keras.metrics.Mean(name="reconstruction_loss")

        # -- Encoder
        inputs = tf.keras.layers.Input(shape=(timesteps, original_dim,))

        h_encoder = RNN(hidden_dim, return_sequences=True)(inputs)

        z_mean = RNN(latent_dim)(h_encoder)
        z_logvar = RNN(latent_dim)(h_encoder)
        z = tf.keras.layers.Lambda(sampling)([z_mean, z_logvar])

        self.encoder = tf.keras.Model(inputs=[inputs], outputs=[z_mean, z_logvar, z], name="Encoder")

        # -- Decoder
        z_inputs = tf.keras.layers.Input(shape=(latent_dim,))

        h_decoder = tf.keras.layers.RepeatVector(timesteps)(z_inputs)
        h_decoder = RNN(hidden_dim, return_sequences=True)(h_decoder)

        outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(original_dim))(h_decoder)

        self.decoder = tf.keras.Model(inputs=[z_inputs], outputs=[outputs], name="Decoder")

    def call(self, x):
        z_mean, z_logvar, z = self.encoder(x)
        x_hat = self.decoder(z, True)
        return x_hat

    @tf.function
    def train_step(self, x):
        if isinstance(x, tuple):
            x = x[0]

        with tf.GradientTape() as tape:
            z_mean, z_logvar, z = self.encode(x)
            x_logit = self.decode(z)

            x_ent = reconstruction_loss(x_logit, x)
            kld = kl_loss(z_mean, z_logvar)
            elbo = x_ent + kld

        gradients = tape.gradient(elbo, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.reconstruction_loss.update_state(x_ent)
        self.kl_loss.update_state(kld)
        self.elbo.update_state(-elbo)

        return {"elbo": self.elbo.result(),
                "reconstruction_loss": self.reconstruction_loss.result(),
                "kl_loss": self.kl_loss.result()}

    @tf.function
    def test_step(self, x):
        if isinstance(x, tuple):
            x = x[0]

        z_mean, z_logvar, z = self.encode(x)
        x_logit = self.decode(z)

        x_ent = reconstruction_loss(x_logit, x)
        kld = kl_loss(z_mean, z_logvar)
        elbo = x_ent + kld

        self.reconstruction_loss.update_state(x_ent)
        self.kl_loss.update_state(kld)
        self.elbo.update_state(-elbo)

        return {"elbo": self.elbo.result(),
                "reconstruction_loss": self.reconstruction_loss.result(),
                "kl_loss": self.kl_loss.result()}

    def encode(self, x):
        """ Encodes multiple samples. """
        z_mean, z_logvar, z = self.encoder(x)
        return z_mean, z_logvar, z

    def decode(self, z, apply_softmax=False):
        """ Decodes multiple latent variables. """
        x_hat = self.decoder(z)
        if apply_softmax:
            return tf.nn.softmax(x_hat, axis=-1)
        return x_hat
