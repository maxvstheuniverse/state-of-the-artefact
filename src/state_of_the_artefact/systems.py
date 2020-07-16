import numpy as np
import random

from state_of_the_artefact.ConceptualSpace import ConceptualSpace
from state_of_the_artefact.utilities import make_onehot

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


class Culture(ConceptualSpace):
    def __init__(self, culture_id, n_agents, n_artefacts, seed, **kwargs):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = f"culture_{culture_id}"
        self.id = culture_id

        self.seed = seed

        self.agents = [Agent(i, culture_id) for i in range(n_agents)]
        self.selected = [np.array(random.choices(self.seed, k=n_artefacts)) for _ in range(n_agents)]

        for agent in self.agents:
            agent.fit(self.seed, batch_size=BATCH_SIZE)

        self.fit(self.seed, batch_size=BATCH_SIZE)

    def select(self, agent_id):
        """ Returns all known arte facts created by the specified agent. """
        return [artefact[4] for artefact in self.repository if artefact[1] == agent_id]


class Agent(ConceptualSpace):
    def __init__(self, agent_id, culture_id, **kwargs):
        super().__init__(TIMESTEPS, DIMENSIONS)
        self.name = f"agent_{culture_id}_{agent_id}"
        self.id = agent_id

    def build(self, z):
        """ Returns a reconstructed artefact for the given latent variables. """
        new_artefact = self.rvae.decode(z).numpy()
        z_mean = self.rvae.encode(new_artefact)
        return make_onehot(new_artefact), z_mean

