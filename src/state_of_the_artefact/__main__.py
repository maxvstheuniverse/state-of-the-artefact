import argparse
import os
import random
import numpy as np
import time

from state_of_the_artefact.Culture import Culture
from state_of_the_artefact.representation import generate_midi_data, create_ctable

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


# def init(args):
#     """ Initializes the levels and seeds the individuals. """

#     characters = [str(c) for c in range(24, 36)]
#     encode, decode = create_ctable(characters)
#     seed = vectorize(generate_midi_data(10000, 10, midi_numbers=range(24, 36)), encode)

#     observer = Observer()
#     cultures = [Culture(i, seed) for i in range(args.num_cultures)]

#     return observer, cultures


def run(args):
    start_time = time.time()
    t = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime())

    # setup the representation
    characters = [str(c) for c in range(24, 36)]
    encode, decode, vectorize = create_ctable(characters)

    seed_path = os.path.join(os.getcwd(), "data", "seeds", "seed.npy")

    if os.path.exists(seed_path):
        print("Loading seed...", end=" ")
        seed = np.load(seed_path)
        print("Done.")
    else:
        print('Generating seed...', end=" ")
        seed = vectorize(generate_midi_data(50000, 10, midi_numbers=range(24, 36)))
        print("Saving.", end=" ")
        np.save(seed_path, seed)
        print("Done.")

    # setup the systems
    # observer = Observer()
    culture = Culture(0, seed, args.n_agents)

    # setup the loop
    selected = np.array([random.choices(seed, k=args.n_artefacts) for _ in range(args.n_agents)])

    print('-' * 80)
    # -- run the loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch:03d}...", end=("\r" if epoch < args.epochs - 1 else " "))

        # INDIVIDUAL –– learn and create

        new_artefacts = []

        for i, agent in enumerate(culture.agents):
            # perception, and find get the agents ideal (the mean)
            ideal = agent.learn(selected[i], decode)

            # build new ideal artefacts
            new_artefact = decode(agent.build(ideal)[0])

            # add to culture, and save ideals for culture to learn the agent's positions
            culture.add(epoch, agent.id, culture.id, new_artefact)
            new_artefacts.append(new_artefact)

        # FIELD -- interact and select
        positions = culture.learn(vectorize(new_artefacts), decode)
        # TODO: calculate area covered, after addition of new artefacts

        for i, agent in enumerate(culture.agents):
            # calcualte distances to other agents
            distances = np.linalg.norm(positions[i] - positions, axis=1)

            # get neighbours, sort, return indices, skip first as this is the current agent.
            neighbours = np.argsort(distances)[1:args.n_neighbours + 1]

            # gather artefacts created by the field
            possible_artefacts = culture.select(agent.id)

            for neighbour in neighbours:
                possible_artefacts += culture.select(neighbour)

            # TODO: draw from a gaussian distribution
            # TODO: with every round reweight artefacts, new ones are more likely?=
            selected[i] = vectorize(random.choices(possible_artefacts, k=args.n_artefacts))

    print("Done.")
    print(f"Time Elapsed: {time.time() - start_time:0.3f}s")

    data, validation = culture.visualize(vectorize)
    data_path = os.path.join(os.getcwd(), "data", "output", f"output_{t}")
    np.save(data_path, np.array(data))
    np.save(data_path + "_validation", validation)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10, dest="epochs",
                        help="The number of rounds for the simulation. Default: 10")

    parser.add_argument("-c", "--cultures", type=int, default=1, dest="n_cultures",
                        help="The number of agents in the simulation. Default: 1")
    parser.add_argument("-a", "--agents", type=int, default=4, dest="n_agents",
                        help="The number of agents in the simulation. Default: 4")
    parser.add_argument("-n", "--neighbours", type=int, default=2, dest="n_neighbours",
                        help="The number of agents selected to be the field. Default: 2")

    parser.add_argument("-i", "--init-epochs", type=int, default=50, dest="init_epochs",
                        help="The number of epochs used for schooling the individuals. Default: 50")
    parser.add_argument("-s", "--artefacts", type=int, default=10, dest="n_artefacts",
                        help="The number of item selected each round. Default: 10")
    args = parser.parse_args()

    # launch simulation
    run(args)
    return 0


if __name__ == "__main__":
    main()
