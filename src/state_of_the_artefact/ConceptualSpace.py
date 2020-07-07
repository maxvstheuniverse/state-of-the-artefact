import tensorflow as tf
import numpy as np
import os
import pandas as pd

from state_of_the_artefact.RVAE_ import RecurrentVariationalAutoEncoder
from state_of_the_artefact.utilities import reverse_sequences


class ConceptualSpace():
    def __init__(self, timesteps, dimensions):
        self.rvae = RecurrentVariationalAutoEncoder(timesteps,
                                                    dimensions[0], dimensions[1], dimensions[2])
        self.rvae.compile(optimizer='adam', loss='categorical_crossentropy')

        self.budget = 200
        self.repository = []

    def fit(self, seed, epochs=25, **kwargs):
        """ Initialize conceptual space with starting seed. """
        model_path = os.path.join(os.getcwd(), "data", "models", f"{self.name}.h5")

        if os.path.exists(model_path):
            print(f"Loading weights for {self.name}...", end=" ")
            self.rvae.load_weights(model_path)
            print("Done.")
        else:
            self.rvae.fit(reverse_sequences(seed), seed, epochs=epochs, **kwargs)
            self.rvae.save_weights(model_path)
            print("Done.")

    def learn(self, artefacts, apply_mean=True):
        """ Trains the individual to understand the presented artefacts.

            Returns the mean of those prestented artefacts.
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

        z_mean, z_logvar, z = self.rvae.encode(x)

        # NOTE: do I take the z or the z_mean variables?
        if apply_mean:
            return z.numpy().mean(axis=0, keepdims=True)
        else:
            return z.numpy()

    def encode(self, artefacts):
        x = reverse_sequences(artefacts)
        return self.rvae.encode(x)

    def store(self, entry):
        self.repository.append(entry)

    def evaluate(self, val_seed):
        """ Evaluate the current state of the conceptual space against a validation seed. """
        # TODO: optimization reverse sequences at initialization
        evaluation = self.rvae.evaluate(reverse_sequences(val_seed), val_seed, verbose=0)
        return evaluation

    def reconstruct(self):
        """ Reconstruct all created artefacts by the agent. """
        _, _, _, artefact_ids, artefacts, o_z_mean = list(zip(*self.repository))

        z_mean, z_logvar, z = self.encode(np.array(artefacts))
        return np.array([*zip(artefact_ids, o_z_mean, z_mean, z_logvar, z)])

    def export(self):
        epochs, agent_ids, culture_ids, artefact_id, artefacts, o_z_mean = list(zip(*self.repository))

        z_mean, z_logvar, z = self.encode(artefacts)

        data = list(zip(epochs, agent_ids, culture_ids, artefact_id, artefacts,
                        o_z_mean, z_mean.numpy(), z_logvar.numpy(), z.numpy()))
        return np.array(data)

