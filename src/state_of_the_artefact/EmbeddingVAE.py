import tensorflow as tf


def reparameterize(args):
    z_mean, z_logvar = args
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return eps * tf.exp(z_logvar * .5) + z_mean


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def kl_loss(mean, logvar):
    return -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))


class EncoderRNN(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, latent_size, name="EncoderRNN", **kwargs):
        super(EncoderRNN, self).__init__(name=name, **kwargs)

        self.embedded = tf.keras.layers.Embedding(108, hidden_size, input_length=input_size)
        self.hidden = tf.keras.layers.LSTM(hidden_size)

        self.dense_mean = tf.keras.layers.Dense(latent_size)
        self.dense_logvar = tf.keras.layers.Dense(latent_size)
        self.sampling = tf.keras.layers.Lambda(reparameterize)

    def call(self, x):
        embedded = self.embedded(x)
        output = self.hidden(embedded)

        z_mean = self.dense_mean(output)
        z_logvar = self.dense_logvar(output)
        z = self.sampling((z_mean, z_logvar))

        return z_mean, z_logvar, z


class DecoderRNN(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, latent_size, name="EncoderRNN", **kwargs):
        super(DecoderRNN, self).__init__(name=name, **kwargs)

        self.repeat = tf.keras.layers.RepeatVector(input_size, input_shape=(latent_size,))
        # self.hidden = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        # self.td_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_size, activation="softmax"))
        self.hidden = tf.keras.layers.LSTM(hidden_size)
        self.td_dense = tf.keras.layers.Dense(input_size, activation="relu")

    @tf.function
    def call(self, z):
        repeat = self.repeat(z)
        hidden = self.hidden(repeat)
        output = self.td_dense(hidden)
        return output


class EmbeddingVAE(tf.keras.Model):
    def __init__(self, input_size, hidden_size, latent_size, name="TextVAE", **kwargs):
        super(EmbeddingVAE, self).__init__(name=name, **kwargs)
        self.encoder = EncoderRNN(input_size, hidden_size, latent_size)
        self.decoder = DecoderRNN(input_size, hidden_size, latent_size)

        self.kl_loss = tf.keras.metrics.Mean(name="kl_loss")

    def encode(self, x):
        z_mean, z_logvar, z = self.encoder(x)
        return z_mean, z_logvar, z

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        z_mean, z_logvar, z = self.encode(x)
        reconstructed = self.decode(z)

        KL = kl_loss(z_mean, z_logvar)
        self.add_loss(KL)
        self.add_metric(self.kl_loss(KL))

        return reconstructed
