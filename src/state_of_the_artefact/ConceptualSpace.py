import tensorflow as tf
import numpy as np
import os
import pandas as pd

from state_of_the_artefact.RVAE import RecurrentVariationalAutoEncoder
from state_of_the_artefact.utilities import reverse_sequences
from state_of_the_artefact.callbacks import KLAnnealing


class ConceptualSpace():
    def __init__(self, timesteps, dimensions):
        original_dim, hidden_dim, latent_dim = dimensions

        self.rvae = RecurrentVariationalAutoEncoder(timesteps,
                                                    original_dim,
                                                    hidden_dim,
                                                    latent_dim)
        self.rvae.compile(optimizer='adam')

        self.budget = 100
        self.repository = []

    def fit(self, seed, epochs=500, annealing_epochs=15, model_path=None, **kwargs):
        """ Initialize conceptual space with starting seed. """
        if model_path is None:
            model_path = os.path.join(os.getcwd(), "data", "models")

        # append model name
        model_path = os.path.join(model_path, f"{self.name}_{epochs}.h5")

        if os.path.exists(model_path):
            print(f"Loading weights for {self.name}...", end=" ")
            self.rvae.load_weights(model_path)
            print("Done.")
        else:
            self.rvae.fit(reverse_sequences(seed), seed,
                          epochs=epochs, callbacks=[KLAnnealing(annealing_epochs)],
                          **kwargs)
            self.rvae.save_weights(model_path)
            print("Done.")

    def learn(self, artefacts, apply_mean=True, from_sample=False):
        """ Trains the individual to understand the presented artefacts.

            If `apply_mean` is `True` it returns the interpolated latent variables of the
            presented artefacts.

            If `from_sample` is `True` it returns an interpolated latent variable sampled
            from a normal distrubution with the mean and variance of the latent variables
            of the learned artefacts.

            Otherwise, it returns the sampled latent variables of each artefact  .
        """
        budget = self.budget
        num_artefacts = len(artefacts)

        x = reverse_sequences(artefacts)

        correct = 0
        while correct < num_artefacts and budget > 0:
            correct = 0
            self.rvae.fit(x, artefacts, epochs=1, batch_size=num_artefacts, verbose=0)
            _, _, z = self.rvae.encode(x)
            reconstructions = self.rvae.decode(z).numpy()

            for artefact, reconstruction in zip(artefacts, reconstructions):
                a = np.argmax(artefact, axis=1)
                b = np.argmax(reconstruction, axis=1)

                if np.array_equiv(a, b):
                    correct += 1

            budget -= 1

        _, _, z = self.rvae.encode(x)
        return z.numpy()

    def encode(self, artefacts):
        x = reverse_sequences(artefacts)
        z_mean, z_logvar, z = self.rvae.encode(x)
        return z_mean.numpy(), z_logvar.numpy(), z.numpy()

    def reconstruct(self):
        """ Reconstruct all created artefacts by the agent. """
        _, _, _, artefact_ids, artefacts, o_z_mean = list(zip(*self.repository))

        z_mean, z_logvar, z = self.encode(np.array(artefacts))
        x_hat = self.rvae.decode(z).numpy()
        return np.array([*zip(artefact_ids, o_z_mean, z_mean, z_logvar, z, artefacts, x_hat)])
