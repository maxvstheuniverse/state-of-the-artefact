import argparse

from state_of_the_artefact.systems import Observer, Culture, Agent
from state_of_the_artefact.representation import generate_midi_data, vectorize, create_ctable

"""
____________Model Outline____________

    ----> External Observer
    |             ¦
    |  -> Culture Space (Domain)       Seed
    |  |          |
    |  |          |
    |  |    Individual Space           Perception, Estimation
    |  |          |
    |  |          |
    |  |   Social Space (Field)        Selection, Evaluation
    |  |          |
    |  |          |
    |  |- Updated Culture Spaces
    |             ¦
    |-- Updated External Observer

____________________________________
"""


def init(args):
    """ Initializes the levels and seeds the individuals. """

    characters = [str(c) for c in range(24, 36)]
    encode, decode = create_ctable(characters)
    seed = vectorize(generate_midi_data(10000, 10, midi_numbers=range(24, 36)), encode)

    observer = Observer()
    cultures = [Culture(i, seed) for i in range(args.num_cultures)]

    return observer, cultures


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rounds", type=int, default=10, dest="num_rounds",
                        help="The number of rounds for the simulation. Default: 10")

    parser.add_argument("-a", "--agents", type=int, default=4, dest="num_agents",
                        help="The number of agents in the simulation. Default: 4")
    parser.add_argument("-c", "--cultures", type=int, default=1, dest="num_cultures",
                        help="The number of agents in the simulation. Default: 1")

    parser.add_argument("-e", "--epochs", type=int, default=50, dest="epochs",
                        help="The number of epochs used for schooling the individuals. Default: 50")
    parser.add_argument("-s", "--artefacts", type=int, default=10, dest="num_artefacts",
                        help="The number of item selected each round. Default: 10")
    args = parser.parse_args()

    # -- Seed Culture, ie generate dataset with a subset of notes
    observer, agents, seed = init(args)
    selected = random.choices(seed, k=args.num_artefacts)

    for r in range(args.num_rounds):
        new_artefacts = []

        # -- perception. Understand and perceive selected artefacts. Learn new ideal?

        for culture in cultures:
            for agent in culture.agents:
                ideal = agent.learn(selected)

                # -- create
                new_artefact = agent.build(ideal)

                # -- store
                new_artefacts.append(new_artefact)
                observer.append(agent.id, culture.id, new_artefact)
                culture.append(agent.id, culture.id, new_artefact)

    return 0
