import numpy as np
import tensorflow as tf
import os

from state_of_the_artefact.config import timesteps, original_dim, hidden_dim, latent_dim
from state_of_the_artefact.Agent import Agent
from state_of_the_artefact.RVAE import RecurrentVariationalAutoEncoder
from state_of_the_artefact.utilities import reverse_sequences


class Culture():
    def __init__(self, identifier, seed, n_agents, learning_budget=100):
        self.id = identifier

        split_at = len(seed) - len(seed) // 10
        self.seed = seed[:split_at]
        self.seed_val = seed[split_at:]

        self.y_seed = reverse_sequences(seed)
        self.budget = learning_budget
        self.agents = [Agent(i, identifier) for i in range(n_agents)]
        self.repository = []
        self.evaluation = []

        self.cs = RecurrentVariationalAutoEncoder(timesteps,
                                                  original_dim, hidden_dim, latent_dim,
                                                  RNN=tf.keras.layers.LSTM)
        self.cs.compile(optimizer="adam", loss="categorical_crossentropy")

        # -- intialize the culture and school the agents
        self.fit(seed, batch_size=32)

        for agent in self.agents:
            agent.fit(self.seed, batch_size=32)

    def fit(self, artefacts, epochs=25, **kwargs):
        """ School the agent, by presenting artefacts that are the starting point of the culture. """
        model_path = os.path.join(os.getcwd(), "data", "models", f"culture_{self.id}.h5")

        if os.path.exists(model_path):
            print(f"Loading weights for culture_{self.id}...", end=" ")
            self.cs.load_weights(model_path)
            print(f"Done.")
        else:
            self.cs.fit(reverse_sequences(artefacts), artefacts, epochs=epochs, **kwargs)
            self.cs.save_weights(model_path)

    def add(self, epoch, agent_id, culture_id, artefact):
        _artefact = [epoch, agent_id, culture_id, artefact]
        self.repository.append(_artefact)

    def select(self, agent_id):
        """ Returns all known artefacts created by the specified agent. """
        return [artefact[3] for artefact in self.repository if artefact[1] == agent_id]

    def learn(self, artefacts, decode_fn):
        """ Trains the individual to understand the presented artefacts """
        # self.artefacts.append(artefacts)

        budget = self.budget
        x = reverse_sequences(artefacts)

        missclassifieds = 1
        while missclassifieds > 0 and budget > 0:
            missclassifieds = 0
            history = self.cs.fit(x, artefacts, epochs=1, batch_size=1, verbose=0)

            reconstructions = self.cs.predict(x)

            for original, reconstruction in zip(artefacts, reconstructions):
                if not np.array_equiv(decode_fn(original), decode_fn(reconstruction)):
                    missclassifieds += 1

            budget -= 1

        # TODO: Add evaluation on the seed.
        _, _, z = self.cs.encode(x)
        return z.numpy()

    def evaluate(self):
        loss, kl_loss = self.cs.evaluate(self.seed_val, verbose=0)
        self.evaluation.append(loss)

    def visualize(self, vectorize_fn):
        round_ids, agent_ids, culture_ids, artefacts = tuple(zip(*self.repository))

        x = reverse_sequences(vectorize_fn(artefacts))
        mean, logvar, z = self.cs.encode(x)

        cs = list(zip(round_ids, agent_ids, culture_ids, mean.numpy(), logvar.numpy(), z.numpy()))
        evaluations = np.array([self.evaluation, *[agent.evaluation for agent in self.agents]])
        return cs, evaluations
