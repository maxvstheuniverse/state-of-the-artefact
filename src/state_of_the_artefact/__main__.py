import argparse
import os
import random
import numpy as np
import pandas as pd
import time
import shortid

from state_of_the_artefact.experiments.two_agents import two_agents
from state_of_the_artefact.systems import Observer, Recommender, Agent
from state_of_the_artefact.representation import generate_midi_data, create_ctable


BATCH_SIZE = 32
TIMESTEPS = 16
MIDI_RANGE = range(24, 36)
SAMPLES = 50000

sid = shortid.ShortId()
data_path = os.path.join(os.getcwd(), "data")


def is_pd(v):
    return isinstance(v, pd.DataFrame) or isinstance(v, pd.Series)


def make_entry(epoch, agent, artefact, z_mean):
    return {"epoch": epoch,
            "id": sid.generate(),
            "agent_id": agent.id,
            "artefact": artefact,
            "z_mean": z_mean}


def simulation_static(args):
    start_time = time.time()
    t = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime())

    metrics = {"args": pd.Series(vars(args)),
               "evaluations": [],
               "reconstructions": [],
               "interactions": []}

    # -----------------------------------------------------------------------------------------
    # -- SETUP

    # -- init domain
    domain_seed_path = os.path.join(data_path, "seeds", f"domain_{SAMPLES}.npy")

    if os.path.exists(domain_seed_path):
        domain_seed = np.load(domain_seed_path)
    else:
        domain_seed = generate_midi_data(SAMPLES, TIMESTEPS, midi_range=MIDI_RANGE)

    recommender = Recommender(domain_seed)

    if not os.path.exists(domain_seed_path):
        np.save(domain_seed_path, domain_seed)  # only save if recommender is done training.

    # -- init agents
    agent_seed_path = os.path.join(data_path, "seeds", f"agent_seeds.npy")

    if os.path.exists(agent_seed_path):
        agent_seeds = np.load(agent_seed_path)
    else:
        agent_seeds = recommender.generate_seeds(args.n_agents, 2000)

    agents = [Agent(i, seed) for i, seed in enumerate(agent_seeds)]

    if not os.path.exists(agent_seed_path):
        np.save(agent_seed_path, agent_seeds)  # only save if agents are done training.

    # -- init artefacts
    selected = [np.array(random.choices(seed, k=args.n_artefacts)) for seed in agent_seeds]

    # -- prepare initial artefacts
    for agent, starters in zip(agents, selected):
        z_means, _, _ = agent.encode(starters)
        initial_entries = [make_entry(-1, agent, artefact, z_mean)
                           for artefact, z_mean in zip(starters, z_means.numpy())]
        agent.save(initial_entries)
        recommender.save(initial_entries)

    # -----------------------------------------------------------------------------------------
    # -- LOOP

    for epoch in range(args.epochs):
        print(f"Epoch {epoch:03d}...", end=" ")
        epoch_start = time.time()

        # -------------------------------------------------------------------------------------
        # -- INDIVIDUALS

        new_entries = []
        evaluation = {}
        reconstruction = {}

        for i, agent in enumerate(agents):
            # learn
            z = agent.learn(selected[i])

            # build
            artefact, z_mean = agent.build(z)
            entry = make_entry(epoch, agent, artefact[0], z_mean)

            # store
            agent.save([entry])
            new_entries.append(entry)

            # NOTE: considerable computational resources
            metrics["evaluations"] += [{"epoch": epoch,
                                        "agent_id": agent.id,
                                        **agent.evaluate("repository")}]

            # NOTE: considerable computational resources
            metrics["reconstructions"] += agent.reconstruct(epoch)

        # -------------------------------------------------------------------------------------
        # -- DOMAIN

        positions = recommender.find_positions(new_entries, save_entries=True)

        # -------------------------------------------------------------------------------------
        # -- FIELD

        for i, agent in enumerate(agents):
            # calculate distances to other agents
            distances = np.linalg.norm(positions[i] - positions, axis=1)

            # get neighbours, sort, return indices, skip first as this is the current agent.
            neighbours = np.argsort(distances)[1:args.n_neighbours + 1]

            # record interactions
            metrics["interactions"] += [{"epoch": epoch,
                                         "agent_id": agent.id,
                                         "position": positions[i],
                                         "neighbours": neighbours,
                                         "distances": distances}]

            # gather artefacts created by the field
            available_artefacts = recommender.select_artefacts(agent.id)

            for neighbour_id in neighbours:
                available_artefacts += recommender.select_artefacts(neighbour_id)

            # uniform selection is a field without ideology.
            selected[i] = np.array(random.choices(available_artefacts, k=args.n_artefacts))

        print(f"{time.time() - epoch_start:0.3f}s", end=("\r"
                                                         if epoch < args.epochs - 1
                                                         else "\n"))

    print(f"Total time elapsed: {time.time() - start_time:0.3f}s")

    # -----------------------------------------------------------------------------------------
    # -- PREPARE FOR EXPORT..

    print("Preparing...", end=" ")

    metrics["domain"] = recommender.export()

    data = pd.Series({k: v if is_pd(v) else pd.DataFrame(v) for k, v in metrics.items()})

    # -----------------------------------------------------------------------------------------
    # -- ..AND SAVE

    print("Saving...", end=" ")

    file_name = f"recommender_a{args.n_agents}_e{args.epochs}_{t}.gzip"
    data.to_pickle(os.path.join(data_path, "output", file_name))

    print("Done.")
    return 0


def simulation_dynamic(args):
    start_time = time.time()
    t = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime())

    metrics = {"args": pd.Series(vars(args)),
               "evaluations": [],
               "reconstructions": [],
               "interactions": []}

    # setup the representation
    seed_paths = [os.path.join(os.getcwd(), "data", "seeds", f"culture_{n}.npy")
                  for n in range(args.n_cultures)]
    seeds = []

    for seed_path in seed_paths:
        if os.path.exists(seed_path):
            print("Loading seed...", end=" ")
            seeds.append(np.load(seed_path))
            print("Done.")
        else:
            print('Generating seed...', end=" ")
            seed = generate_midi_data(1500 * BATCH_SIZE, TIMESTEPS, midi_range=MIDI_RANGE)
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
                ideal = agent.learn(culture.selected[j], apply_mean=True)

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
                available_artefacts = culture.select(agent.id)

                # then append the neighbours
                for neighbour in neighbours:
                    available_artefacts += culture.select(neighbour)

                # TODO: draw from a gaussian distribution or FRECENCY
                # TODO: with every round reweight artefacts, new ones are more likely?
                # TODO: use a novelty measure for selecting artefacts, the most novel ones have the highst prob?
                culture.selected[j] = np.array(random.choices(available_artefacts, k=args.n_artefacts))

        evaluations.append(evaluation)


    print("Done.")
    print(f"Time Elapsed: {time.time() - start_time:0.3f}s")

    # -----------------------------------------------------------------------------------------
    # -- PREPARE FOR EXPORT..

    print("Preparing...", end=" ")

    metrics["domain"] = recommender.export()

    data = pd.Series({k: v if is_pd(v) else pd.DataFrame(v) for k, v in metrics.items()})

    # -----------------------------------------------------------------------------------------
    # -- ..AND SAVE

    print("Saving...", end=" ")

    data.to_pickle(os.path.join(data_path, f"a{args.n_agents}_e{ars.epochs}_{t}.gzip"))

    print("Done.")
    return 0

    # # -- still dealing with a single culture
    # data = observer.cultures[0].export()
    # data_path = os.path.join(os.getcwd(), "data", "output", f"500_output_{t}")

    # np.save(data_path + ".npy", data)
    # np.save(data_path + "_evaluations.npy", np.array(evaluations))
    # np.save(data_path + "_reconstructions.npy", np.array(reconstructions))
    # np.save(data_path + "_interactions.npy", np.array(agent_interactions))


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10, dest="epochs",
                        help="The number of rounds for the simulation. Default: 10")

    parser.add_argument("-c", "--cultures", type=int, default=1, dest="n_cultures",
                        help="The number of agents in the simulation. Default: 1")
    parser.add_argument("-a", "--agents", type=int, default=4, dest="n_agents",
                        help="The number of agents in the simulation. Default: 4")
    parser.add_argument("-n", "--neighbours", type=int, default=1, dest="n_neighbours",
                        help="The number of agents selected to be the field. Default: 2")

    parser.add_argument("-i", "--init-epochs", type=int, default=250, dest="init_epochs",
                        help="The number of epochs used for schooling the individuals. \
                              Default: 250")
    parser.add_argument("-s", "--artefacts", type=int, default=10, dest="n_artefacts",
                        help="The number of item selected each round. Default: 10")

    parser.add_argument("--from_sample", type=bool, default=False, dest="from_sample",
                        help="If `True` the agent returns a sampled position. \
                              When `False` it returns the mean.")
    parser.add_argument("--novelty", type=float, default=1.0, dest="novelty",
                        help="Sets the novelty preference for the agents.")
    args = parser.parse_args()

    # launch simulation
    simulation_static(args)
    return 0


if __name__ == "__main__":
    main()
