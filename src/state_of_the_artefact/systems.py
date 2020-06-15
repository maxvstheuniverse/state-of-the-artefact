import numpy as np

from tensorflow import keras

from state_of_the_artefact.RVAE import RecurrentVariationalAutoEncoder
from state_of_the_artefact.utilities import reverse_sequences

class ConceptualSpace():
    def __init__(self, timesteps, original_dim, hidden_dim, latent_dim):
        self.artefacts = []

        self.optimizer = keras.optimizers.Adam(1e-3)
        self.rvae = RecurrentVariationalAutoEncoder(timesteps,
                                                    original_dim, hidden_dim, latent_dim,
                                                    RNN=keras.layers.LSTM)
        self.rvae.compile(optimizer=self.optimizer, loss="categorical_crossentropy")


class Observer():
    def __init__(self):
        super()
        self.artefacts = []
        self.cs = ConceptualSpace(10, 12, 128, 32)

    def append(self, round_id, agent_id, culture_id, artefact):
        _artefact = [round_id, agend_id, culture_id, artefact]
        self.artefacts.append(_artefact)

    def visualize(self):
        round_ids, agent_ids, culture_ids, artefacts = tuple(zip(*artefacts))
        x = reverse_sequences(artefacts)

        cs = ConceptualSpace(10, 12, 128, 32)
        cs.rvae.fit(x, y, epochs=100, batch_size=128)

        return zip(round_ids, agent_ids, culture_ids, cs.rvae.predict(x))


class Culture():
    def __init__(self, identifier, seed):
        super()
        self.id = identifier
        self.seed = seed
        self.agents = [Agent((identifier, i)) for i in range(args.num_agents)]
        self.artefacts = []

        # -- and school the agents
        for agent in self.agents:
            agent.school(self.seed, batch_size=128)

    def append(self, round_id, agent_id, culture_id, artefact):
        _artefact = [round_id, agend_id, culture_id, artefact]
        self.artefacts.append(_artefact)

    def visualize(self):
        round_ids, agent_ids, culture_ids, artefacts = tuple(zip(*artefacts))
        x = reverse_sequences(artefacts)

        cs = ConceptualSpace(10, 12, 128, 32)
        cs.rvae.fit(x, y, epochs=100, batch_size=128)

        return zip(round_ids, agent_ids, culture_ids, cs.rvae.predict(x))


class Agent():
    def __init__(self, identifier, budget=100):
        self.id = identifier
        self.budget = budget
        self.cs = ConceptualSpace(10, 12, 128, 32)

    def school(self, artefacts, epochs=50, **kwargs):
        """ School the agent, by presenting artefacts that are the starting point of the culture. """
        x = reverse_sequences(artefacts)
        history = self.cs.rvae.fit(x, artefacts, epochs=epochs, **kwargs)
        return history

    def learn(self, artefacts):
        """ Trains the individual to understand the presented artefacts """
        self.cs.artefacts.append(artefacts)  # append artefacts to the conceptual space repository.

        budget = self.budget
        x = reverse_sequences(artefacts)

        while missclassifieds > 0 or budget > 0:
            missclassifieds = 0
            history = self.cs.rvae.fit(x, artefacts, epochs=1, verbose=0)

            reconstructions = self.cs.rvae.predict(x)

            for original, reconstruction in zip(artefacts, reconstructions):
                if not np.array_equiv(original, reconstruction):
                    misclassifieds += 1

            budget -= 1

        _, _, z = self.rvae.encode(x)
        return z.numpy().mean(axis=0, keepdims=True)

    def build(self, z):
        return self.rvae.decode(z).numpy()

