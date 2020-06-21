import numpy as np
import tensorflow as tf
import os

from state_of_the_artefact.config import timesteps, original_dim, hidden_dim, latent_dim
from state_of_the_artefact.RVAE import RecurrentVariationalAutoEncoder
from state_of_the_artefact.utilities import reverse_sequences


class Agent():
    def __init__(self, identifier, culture_id, learn_budget=100):
        self.id = identifier
        self.culture_id = culture_id
        self.budget = learn_budget
        self.validation = []

        self.rvae = RecurrentVariationalAutoEncoder(timesteps,
                                                    original_dim, hidden_dim, latent_dim,
                                                    RNN=tf.keras.layers.LSTM)
        self.rvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy")

    def fit(self, artefacts, epochs=50, **kwargs):
        """ School the agent, by presenting artefacts that are the starting point of the culture. """
        model_path = os.path.join(os.getcwd(), "data", "models", f"agent_{self.culture_id}_{self.id}.h5")

        if os.path.exists(model_path):
            print(f"Loading weights for agent_{self.culture_id}_{self.id}...", end=" ")
            self.rvae.load_weights(model_path)
            print(f"Done.")
        else:
            self.rvae.fit(reverse_sequences(artefacts), artefacts, epochs=epochs, **kwargs)
            self.rvae.save_weights(model_path)

    def learn(self, artefacts, decode_fn):
        """ Trains the individual to understand the presented artefacts """
        budget = self.budget
        x = reverse_sequences(artefacts)

        missclassifieds = 1
        while missclassifieds > 0 and budget > 0:
            missclassifieds = 0
            history = self.rvae.fit(x, artefacts, epochs=1, verbose=0)

            reconstructions = self.rvae.predict(x)

            for original, reconstruction in zip(artefacts, reconstructions):
                if not np.array_equiv(decode_fn(original), decode_fn(reconstruction)):
                    missclassifieds += 1

            budget -= 1

        # -- check validation
        loss = self.rvae.evaluate(self.seed, verbose=0)
        self.validation.append(loss)

        _, _, z = self.rvae.encode(x)
        return z.numpy().mean(axis=0, keepdims=True)

    def build(self, z):
        return self.rvae.decode(z).numpy()
