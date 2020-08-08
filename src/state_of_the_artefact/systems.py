import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.neighbors import BallTree

from state_of_the_artefact.ConceptualSpace import ConceptualSpace
from state_of_the_artefact.utilities import reverse_sequences, kde_density, one_hot

TIMESTEPS = 16
DIMENSIONS = (12, 128, 32)  # input, hidden, latent
BATCH_SIZE = 32


class Recommender(ConceptualSpace):
    """ The Recommender is static system non-learning Conceptual Space used
        for making predictions about individual positions
    """
    def __init__(self, seed, args=None, model_path=None):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = "recommender"
        self.seed = seed
        self.tree = None
        self.frecency = {}

        self.fit(self.seed, epochs=1000, batch_size=64, model_path=model_path)

    def generate_ball_tree(self):
        points = np.array([entry["domain_z_mean"] for entry in self.repository])
        self.tree = BallTree(points)

    def get_frecency_counts(self, artefact_ids):
        return [self.frecency[artefact_id] for artefact_id in artefact_ids]

    def update_frecency(self, artefact_ids):
        # decay, minimum is 1
        frecency = {k: count - 1 if count > 1 else 1
                    for k, count in self.frecency.items()}

        # growth
        for artefact_id in artefact_ids:
            frecency.update({artefact_id: frecency[artefact_id] + 1})

        self.frecency = frecency

    def find_densities(self, ids, method="kde", r=1.0):
        x = np.array([entry['domain_z_mean']
                      for entry in self.repository
                      if entry["id"] in ids])

        if method == "knn":
            pass  # TODO

        if method == "kde":
            return kde_density(x, self.tree)

    def find_positions(self, entries, save_entries=True, frecency=False):
        artefacts = np.array([entry["artefact"] for entry in entries])
        x = reverse_sequences(one_hot(artefacts))
        zs = self.rvae.encode(x)

        if save_entries:
            self.save(entries, zs, frecency=frecency)

        return zs[0].numpy()

    def select_artefacts(self, agent_id, with_ids=False):
        """ NOTE: Artefacts not in one-hot encoding. """
        artefacts = []
        for entry in self.repository:
            if entry["agent_id"] == agent_id:
                if with_ids:
                    artefacts.append((entry["id"], entry["artefact"]))
                else:
                    artefacts.append(entry["artefact"])

        return np.array(artefacts)

    def select_entries(self, agent_id):
        return [entry for entry in self.repository if agent_id == entry["agent_id"]]

    def check_artefact(self, agent_id, artefact):
        """ Check if artefact is already created by the agent. """
        artefact = np.argmax(artefact, axis=-1)
        artefacts = self.select_artefacts(agent_id)
        return np.any(np.equal(a, b).all(axis=1))

    def save(self, entries, zs=None, frecency=False):
        """ Stores entries in the reposity, generates latent encodings if not provided. """
        if zs is None:
            artefacts = np.array([entry["artefact"] for entry in entries])
            x = reverse_sequences(one_hot(artefacts))
            zs = self.rvae.encode(x)

        for entry, z_mean, z_logvar, z in zip(entries, *zs):
            domain_entry = {**entry,
                            "domain_z_mean": z_mean.numpy(),
                            "domain_z_logvar": z_logvar.numpy(),
                            "domain_z": z.numpy()}

            self.repository.append(domain_entry)

            if frecency:
                self.frecency.update({entry['id']: 10})

    def export(self):
        return self.repository


class Agent(ConceptualSpace):
    def __init__(self, agent_id, seed, model_path=None, **kwargs):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = f"agent_{agent_id}"
        self.id = agent_id
        self.fit(seed, epochs=500, batch_size=BATCH_SIZE, model_path=model_path)

    def sample(self, sample_mode="mean", stddev=.25, n_artefacts=10, z_means=None, decode=False):

        if sample_mode == "mean" and z_means is not None:
            means = z_means.mean(axis=0, keepdims=True)
            return np.random.normal(means, stddev, (n_artefacts, 32))

        if sample_mode == "origin":
            return np.random.normal(0.0, stddev, (n_artefacts, 32))

    def build(self, z):
        """ Returns a reconstructed artefact for the given latent variables. """
        artefacts = self.rvae.decode(z, apply_onehot=True)
        z_means, _, _ = self.rvae.encode(artefacts)
        return artefacts, z_means.numpy()

    def evaluate(self, x):
        """ Returns the evaluation and the accuracy for a perfect reconstruction and timestep
            accuracy.

            If `x` is either `"seed"` or `"repository"` the evaluation is based on the agents'
            local data. If `x` is a `ndarray` it creates a reversed version for evaluation.
        """

        x_reverse = reverse_sequences(x)
        evaluation = self.rvae.evaluate(x_reverse, x, verbose=0, batch_size=1, return_dict=True)

        _, _, z = self.rvae.encode(x)
        reconstructions = self.rvae.decode(z)

        correct = 0
        ts_correct = 0

        for original, reconstruction in zip(x, reconstructions):
            a = np.argmax(original, axis=1)
            b = np.argmax(reconstruction, axis=1)

            if np.array_equal(a, b):
                correct += 1

            ts_correct += np.sum([x == y for x, y in zip(a, b)]) / TIMESTEPS

        evaluation["abs_accuracy"] = correct / len(x)
        evaluation["ts_accuracy"] = ts_correct / len(x)
        return evaluation

    def reconstruct(self, epoch, entries, artefacts):
        """ Returns a `list` with the data of all reconstructed artefacts
            and the current epoch.
        """
        x = reverse_sequences(artefacts)
        z_mean, _, z = self.rvae.encode(x)
        x_hat = np.argmax(self.rvae.decode(z), axis=-1)

        reconstructions = [{"rz_mean": rz_mean,
                            "rz": rz,
                            "reconstruction": r}
                           for rz_mean, rz, r in zip(z_mean.numpy(), z.numpy(), x_hat)]

        entries = [{"current_epoch": epoch,
                    **entry,
                    **reconstruction}
                   for entry, reconstruction in zip(entries, reconstructions)]

        return entries


