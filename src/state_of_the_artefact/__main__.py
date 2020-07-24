import argparse
import os
import random
import numpy as np
import time
import shortid


from state_of_the_artefact.systems import Observer
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

BATCH_SIZE = 32
TIMESTEPS = 16
MIDI_RANGE = range(24, 36)

sid = shortid.ShortId()


def run(args):
    start_time = time.time()
    t = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime())

    # setup the representation
    characters = [f"{pitch}" for pitch in MIDI_RANGE]
    encode, decode, vectorize = create_ctable(characters)

    seed_paths = [os.path.join(os.getcwd(), "data", "seeds", f"culture_{n}.npy") for n in range(args.n_cultures)]
    seeds = []

    for seed_path in seed_paths:
        if os.path.exists(seed_path):
            print("Loading seed...", end=" ")
            seeds.append(np.load(seed_path))
            print("Done.")
        else:
            print('Generating seed...', end=" ")
            seed = vectorize(generate_midi_data(1500 * BATCH_SIZE, TIMESTEPS, midi_range=MIDI_RANGE))
            seeds.append(seed)
            print("Saving.", end=" ")
            np.save(seed_path, seed)
            print("Done.")

    # setup the systems
    observer = Observer(args.n_cultures, args.n_agents, args.n_artefacts, seeds)

    # setup the loop
    print('-' * 80)

    evaluations = []
    reconstructions = {'cultures': [], 'agents': []}
    agent_interactions = []

    # -- run the loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch:03d}...", end=("\r" if epoch < args.epochs - 1 else " "))

        evaluation = []

        # -- CULTURES
        for i, culture in enumerate(observer.cultures):

            # INDIVIDUAL –– learn and create
            new_artefacts = []

            for j, agent in enumerate(culture.agents):
                # start the agent off with a basic pool of "seen" artefacts
                if epoch == 0:
                    # take some of the training examples to seed the repositories
                    starters = culture.selected[j]
                    z_means, _, _ = agent.encode(starters)

                    for artefact, z_mean in zip(starters, z_means):
                        entry = [-1, agent.id, culture.id, sid.generate(), artefact, z_mean]
                        culture.store(entry)
                        agent.store(entry)

                # perception, and find get the agents ideal (the mean of z)
                ideal = agent.learn(culture.selected[j], apply_mean=True, from_sample=False)

                # every 10 epochs evaluate
                if epoch % 5 == 0 or epoch == args.epochs - 1:
                    evaluation.append(agent.evaluate(culture.seed))
                    reconstructions['agents'].append(agent.reconstruct())

                # build new ideal artefacts
                new_artefact, z_mean = agent.build(ideal)

                entry = [epoch, agent.id, culture.id, sid.generate(), new_artefact[0], z_mean]

                observer.store(entry)
                culture.store(entry)
                agent.store(entry)

                # and prepare culture for adapting to new artefacts
                new_artefacts += list(new_artefact)

            # FIELD -- interact and select
            positions = culture.learn(np.array(new_artefacts), apply_mean=False)

            # every 10 epochs evaluate
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                evaluation.append(culture.evaluate(culture.seed))
                reconstructions['cultures'].append(culture.reconstruct())

            for j, agent in enumerate(culture.agents):
                # calculate distances to other agents
                distances = np.linalg.norm(positions[j] - positions, axis=1)

                # get neighbours, sort, return indices, skip first as this is the current agent.
                neighbours = np.argsort(distances)[1:args.n_neighbours + 1]
                agent_interactions.append([epoch, agent.id, neighbours])

                # gather artefacts created by the field
                # first the artefact by the current agent
                possible_artefacts = culture.select(agent.id)

                # then append the neighbours
                for neighbour in neighbours:
                    possible_artefacts += culture.select(neighbour)

                # TODO: draw from a gaussian distribution or FRECENCY
                # TODO: with every round reweight artefacts, new ones are more likely?
                # TODO: use a novelty measure for selecting artefacts, the most novel ones have the highst prob?
                culture.selected[j] = np.array(random.choices(possible_artefacts, k=args.n_artefacts))

        evaluations.append(evaluation)

    print("Done.")
    print(f"Time Elapsed: {time.time() - start_time:0.3f}s")

    # -- still dealing with a single culture
    data = observer.cultures[0].export()
    data_path = os.path.join(os.getcwd(), "data", "output", f"500_output_{t}")

    np.save(data_path + ".npy", data)
    np.save(data_path + "_evaluations.npy", np.array(evaluations))
    np.save(data_path + "_reconstructions.npy", np.array(reconstructions))
    np.save(data_path + "_interactions.npy", np.array(agent_interactions))


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
