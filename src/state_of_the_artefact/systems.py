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
    def __init__(self, seed, args=None):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = "recommender"
        self.seed = seed
        self.tree = None
        self.frecency = {}

        self.fit(self.seed, epochs=2000, batch_size=64)

    def generate_seeds(self, n_agents, samples=2000):
        stddev = 4.0 / n_agents
        means = np.linspace(-3.5 + stddev, 3.5 - stddev, num=n_agents)

        zs = [tf.random.normal(shape=(samples, self.rvae.latent_dim),
                               mean=mean, stddev=stddev)
              for mean in means]

        return [self.rvae.decode(z, apply_onehot=True) for z in zs]

    def generate_ball_tree(self):
        points = [entry["domain_z_mean"] for entry in self.repository]
        self.tree = BallTree(points)

    def get_frencency_counts(self, artefact_ids):
        return [self.frecency[artefact_id] for artefact_id in artefact_ids]

    def update_frecency(self, artefact_ids):
        # decay, do not decay below 1
        frecency = {k: count - 1 if count > 1 else 1
                    for k, count in self.frecency.items()}

        # growth
        for artefact_id in artefact_ids:
            frecency[artefact_id] = frecency[artefact_id] + 1

        self.frecency = frecency

    def find_densities(self, x, method="kde", r=1.0):
        if method == "knn":
            pass  # TODO

        if method == "kde":
            return kde_density(x, self.tree)

    def find_positions(self, entries, save_entries=True, frecency=False):
        x = reverse_sequences(one_hot([entry["artefact"] for entry in entries]))
        z_means, z_logvars, zs = self.rvae.encode(x)

        if save_entries:
            self.save(entries, zs, frecency=frecency)

        return z_means.numpy()

    def select_artefacts(self, agent_id, frecency=False):
        """ NOTE: Artefacts not in one-hot encoding. """
        artefacts = []
        for entry in self.repository:
            if entry["agent_id"] == agent_id:
                if frecency:
                    artefacts.append((entry["artefact_id"], entry["artefact"]))
                else:
                    artefacts.append(entry["artefact"])

        return np.array(artefacts)

    def check_artefact(self, agent_id, artefact):
        """ Check if artefact is already created by the agent. """
        artefact = np.argmax(artefact, axis=-1)
        artefacts = self.select_artefacts(agent_id)
        return np.any(np.equal(a, b).all(axis=1))

    def save(self, entries, zs=None, frecency=False):
        """ Stores entries in the reposity, generates latent encodings if not provided. """
        if zs is None:
            x = reverse_sequences(one_hot([[entry["artefact"]] for entry in entries]))
            zs = self.rvae.encode(x)

        for entry, z_mean, z_logvar, z in zip(entries, *zs):
            domain_entry = {**entry,
                            "domain_z_mean": z_mean.numpy(),
                            "domain_z_logvar": z_logvar.numpy(),
                            "domain_z": z.numpy()}

            self.repository.append(domain_entry)

            if frecency:
                self.frecency.append({entry['artefact_id']: 10})

    def export(self):
        return self.repository


class Agent(ConceptualSpace):
    def __init__(self, agent_id, seed, **kwargs):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = f"agent_{agent_id}"  # f"agent_{culture_id}_{agent_id}"
        self.id = agent_id
        self.fit(seed, batch_size=BATCH_SIZE)

    def sample(self, zs, sample_mode="mean"):
        mean = zs.mean(axis=0, keepdims=True)

        if sample_mode == "mean":
            return mean

        if sample_mode == "sample":
            logvar = zs.var(axis=0, keepdims=True)
            return np.random.normal(mean, logvar, (1, 32))

    def build(self, z):
        """ Returns a reconstructed artefact for the given latent variables. """
        artefact = self.rvae.decode(z, apply_onehot=True)
        z_mean, _, _ = self.rvae.encode(artefact)
        return artefact[0], z_mean.numpy()

    def evaluate(self, x):
        """ Returns the evaluation and the accuracy for a perfect reconstruction and timestep
            accuracy.

            If `x` is either `"seed"` or `"repository"` the evaluation is based on the agents'
            local data. If `x` is a `ndarray` it creates a reversed version for evaluation.
        """

        if isinstance(x, np.ndarray):
            x_hat = x
            x = reverse_sequences(x)

        if x == "seed":
            x_hat = self.seed
            x = self.seed_reversed

        if x == "repository":
            x_hat = np.array(self.artefacts)
            x = np.array(self.artefacts_reversed)

        evaluation = self.rvae.evaluate(x, x_hat, verbose=0, batch_size=1, return_dict=True)

        _, _, z = self.rvae.encode(x)
        reconstructions = self.rvae.decode(z)

        correct = 0
        ts_correct = 0

        for original, reconstruction in zip(x, reconstructions):
            a = np.argmax(original, axis=1)
            b = np.argmax(reconstruction, axis=1)

            if np.array_equiv(a, b):
                correct += 1

            ts_correct += np.sum([0 if x == y else 1 for x, y in zip(a, b)]) / TIMESTEPS

        evaluation["accuracy"] = correct / len(x)
        evaluation["ts_accuracy"] = ts_correct / len(x)
        return evaluation

    def reconstruct(self, epoch):
        """ Returns a `list` with the data of all reconstructed artefacts
            and the current epoch.
        """
        z_mean, _, z = self.rvae.encode(np.array(self.artefacts_reversed))
        x_hat = self.rvae.decode(z, apply_softmax=True, apply_onehot=True)

        reconstructions = [{"rz_mean": rz_mean,
                            "rz": rz,
                            "reconstruction": r}
                           for rz_mean, rz, r in zip(z_mean.numpy(), z.numpy(), x_hat)]

        entries = [{"current_epoch": epoch,
                    **entry,
                    **reconstruction}
                   for entry, reconstruction in zip(self.repository, reconstructions)]

        return entries


