import numpy as np
import pandas as pd
import tensorflow as tf
import random

from state_of_the_artefact.ConceptualSpace import ConceptualSpace
from state_of_the_artefact.utilities import reverse_sequences

TIMESTEPS = 16
DIMENSIONS = (12, 128, 32)  # input, hidden, latent
BATCH_SIZE = 32


class Observer(ConceptualSpace):
    def __init__(self, n_cultures, n_agents, n_artefacts, seeds):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = "observer"

        # -- make sure there is a seed for every culture
        assert len(seeds) == n_cultures

        # -- initialize the cultures and its agents
        self.cultures = [Culture(i, n_agents, n_artefacts, seeds[i]) for i in range(n_cultures)]

        # -- initialize the observer
        # self.fit(self.seeds, batch_size=BATCH_SIZE)


class Recommender(ConceptualSpace):
    """ The Recommender is static system non-learning Conceptual Space used
        for making predictions about individual positions
    """
    def __init__(self, seed):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = "recommender"
        self.seed = seed
        self.fit(self.seed, epochs=50, batch_size=BATCH_SIZE)
        self.repository = []

    def generate_seeds(self, n_agents, samples=2000):
        stddev = 1.0 / n_agents
        means = np.linspace(-4.0 + stddev, 4.0 - stddev, num=n_agents)

        zs = [tf.random.normal(shape=(samples, self.rvae.latent_dim),
                               mean=means[i], stddev=stddev)
              for i in range(n_agents)]

        return [self.rvae.decode(z, apply_onehot=True) for z in zs]

    def find_positions(self, entries, save_entries=True):
        x = reverse_sequences([entry["artefact"] for entry in entries])
        z_means, z_logvars, zs = self.rvae.encode(x)

        if save_entries:
            for entry, z_mean, z_logvar, z in zip(entries, z_means, z_logvars, zs):
                # append extra domain information
                domain_entry = {**entry,
                                "domain_z_mean": z_mean.numpy(),
                                "domain_z_logvar": z_logvar.numpy(),
                                "domain_z": z.numpy()}

                # store entry
                self.repository.append(domain_entry)

        return z_means.numpy()

    def select_artefacts(self, agent_id):
        # NOTE: Do I want to keep track of every artefact selected?
        return [entry["artefact"] for entry in self.repository if entry["agent_id"] == agent_id]

    def save(self, entries, evaluate=True):
        # TODO: Refactor
        x = reverse_sequences([entry["artefact"] for entry in entries])
        z_means, z_logvars, zs = self.rvae.encode(x)

        for entry, z_mean, z_logvar, z in zip(entries, z_means, z_logvars, zs):
            # append extra domain information
            domain_entry = {**entry,
                            "domain_z_mean": z_mean.numpy(),
                            "domain_z_logvar": z_logvar.numpy(),
                            "domain_z": z.numpy()}

            # store entry
            self.repository.append(domain_entry)

    def export(self):
        return self.repository


class Culture(ConceptualSpace):
    def __init__(self, culture_id, n_agents, n_artefacts, seed, **kwargs):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = f"culture_{culture_id}"
        self.id = culture_id

        self.seed = seed

        # TODO: Use unique seed for every agent?
        self.agents = [Agent(i, culture_id) for i in range(n_agents)]
        self.selected = [np.array(random.choices(self.seed, k=n_artefacts)) for _ in range(n_agents)]

        for agent in self.agents:
            agent.fit(self.seed, batch_size=BATCH_SIZE)

        self.fit(self.seed, batch_size=BATCH_SIZE)

    def select(self, agent_id):
        """ Returns all known arte facts created by the specified agent. """
        return [artefact[4] for artefact in self.repository if artefact[1] == agent_id]


class Agent(ConceptualSpace):
    def __init__(self, agent_id, seed, **kwargs):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = f"agent_{agent_id}"  # f"agent_{culture_id}_{agent_id}"
        self.id = agent_id

        self.repository = []

        self.artefacts = []
        self.artefacts_reversed = []

        self.seed = seed
        self.seed_reversed = reverse_sequences(seed)

        self.fit(self.seed, batch_size=BATCH_SIZE)

    def select(self):
        # NOTE: Do I want to keep track of every artefact selected?
        return [entry["artefact"] for entry in self.repository]

    def build(self, z):
        """ Returns a reconstructed artefact for the given latent variables. """
        artefact = self.rvae.decode(z, apply_onehot=True)
        z_mean, _, _ = self.rvae.encode(artefact)
        return artefact, z_mean.numpy()

    def save(self, entries):
        for entry in entries:
            self.repository.append(entry)

            # and store the artefact (and its reverse) for evaluation
            # saves some resources by reversing the artefacts each epoch.
            self.artefacts.append(entry["artefact"])
            self.artefacts_reversed.append(entry["artefact"][::-1])

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


