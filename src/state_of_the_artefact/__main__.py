import argparse
import boto3
import numpy as np
import os
import pandas as pd
import shortid
import random
import time

from state_of_the_artefact.helpers.upload import upload_obj
from state_of_the_artefact.representation import generate_midi_data, create_ctable
from state_of_the_artefact.systems import Recommender, Agent
from state_of_the_artefact.utilities import one_hot

"""
TODO:

[x] finalize s3
[ ] test s3 on liacs computers
[/] avoid duplicates
[x] frecency
[ ] novelty in agents
[ ]    - hedonic from center (sample 10 pick highest)
[ ]    - sample many with pick density
[ ] agent preference for novelty (1-5 levels of stddev from the center)
[ ] sampling in agents
[ ] try different scales in the dataset (chromatic, heptatonic, pentatonic, tetratonic)

EXPERIMENTS:

1. vae evaluations
2. cliques
3. rate of saturation, density and volume
4. social interaction (uniform, density, frecency)
5. individual (interpolation, sampling)
6. individual (novelty preference vs no novelty preference)
"""


BATCH_SIZE = 32
TIMESTEPS = 16
MIDI_RANGE = range(24, 36)
DEPTH = len(MIDI_RANGE)
SAMPLES = 20000

sid = shortid.ShortId()
data_path = os.path.join(os.getcwd(), "data")


def is_pd(v):
    return isinstance(v, pd.DataFrame) or isinstance(v, pd.Series)


def make_entry(epoch, agent, artefact, z_mean):
    return {"epoch": epoch,
            "id": sid.generate(),
            "agent_id": agent.id,
            "artefact": np.argmax(artefact, axis=-1),
            "z_mean": z_mean}


def run_simulation(args):
    start_time = time.time()

    data = {"args": pd.Series(vars(args)),
            "evaluations": [],
            "reconstructions": [],
            "interactions": [],
            "domain": []}

    # -----------------------------------------------------------------------------------------
    # -- SETUP

    domain_seed_path = os.path.join(data_path, "seeds", "domain_seed.npy")
    agent_seeds_path = os.path.join(data_path, "seeds", "agent_seeds.npy")

    try:
        domain_seed = np.load(domain_seed_path)
        agent_seeds = np.load(agent_seeds_path)
    except IOError as e:
        print(e.args)
        return 1

    recommender = Recommender(domain_seed)
    agents = [Agent(i, seed) for i, seed in enumerate(agent_seeds)]

    # -- Other sim settings
    frecency = args.interaction_mode == 'frecency'

    # -----------------------------------------------------------------------------------------
    # -- EPOCH 0

    # -- prepare initial artefacts
    # TODO: permutation

    selected = [np.random.choice(available_artefacts,
                                 size=args.n_artefacts, replace=False)
                for seed in agent_seeds]

    # -- make entries for intial artefacts
    for agent, artefacts in zip(agents, selected):
        z_means, _, _ = agent.encode(artefacts)
        initial_entries = [make_entry(0, agent, artefact, z_mean)
                           for artefact, z_mean in zip(artefacts, z_means)]

        recommender.save(initial_entries, frecency)

    if args.interaction_mode == "density":
        # -- generate the first ball tree
        recommender.generate_ball_tree()

    # -- export starting domain
    data["domain"] += [recommender.export()]

    # -----------------------------------------------------------------------------------------
    # -- LOOP

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:03d}...", end=" ")
        epoch_start = time.time()

        # -------------------------------------------------------------------------------------
        # -- INDIVIDUALS

        new_entries = []
        evaluation = {}
        reconstruction = {}

        for i, agent in enumerate(agents):
            # learn
            zs = agent.learn(selected[i])

            tries = 10
            while tries > 0:
                # sample
                z = agent.sample(zs, args.sample_mode)

                #  build, in one-hot encoding
                artefact, z_mean = agent.build(z)

                # check
                if recommender.check_artefact(artefact):
                    entry = make_entry(epoch, agent, artefact, z_mean)

                    # store
                    agent.generate_ball_tree()
                    new_entries.append(entry)
                    break
                else:
                    tries -= 1

            if tries == 0:
                # generate a random sample from the whole!
                pass

            # NOTE: considerable computational resources, consider every n epochs
            data["evaluations"] += [{"epoch": epoch,
                                     "agent_id": agent.id,
                                     **agent.evaluate("repository")}]

            # NOTE: considerable computational resources, consider every n epochs
            data["reconstructions"] += agent.reconstruct(epoch)

        # -------------------------------------------------------------------------------------
        # -- DOMAIN

        positions = recommender.find_positions(new_entries,
                                               save_entries=True,
                                               frecency=frecency)

        # -------------------------------------------------------------------------------------
        # -- FIELD

        selected_ids = []

        # TODO: implement different social interaction policies.
        for i, agent in enumerate(agents):
            # calculate distances to other agents
            distances = np.linalg.norm(positions[i] - positions, axis=1)

            # get neighbours, sort, return indices, skip first as this is the current agent.
            neighbours = np.argsort(distances)[1:args.n_neighbours + 1]

            # record interactions
            data["interactions"] += [{"epoch": epoch,
                                      "agent_id": agent.id,
                                      "position": positions[i],
                                      "neighbours": neighbours,
                                      "distances": distances}]

            # gather artefacts created by the field
            available_artefacts = recommender.select_artefacts(agent.id,
                                                               frecency=frecency)

            for neighbour_id in neighbours:
                neighbour_artefacts = recommender.select_artefacts(neighbour_id,
                                                                   frecency=frecency)
                available_artefacts = np.vstack(available_artefacts, neighbour_artefacts)

            # default is uniform, which means no ideology in the population
            probabilities = None

            # if based on density, the population favors artefacts which are "more"
            # unique, based on the estimated density in their space.
            if args.interaction_mode == "density":
                # using the inverse in order to favor low density artefacts.
                densities = 1 / recommender.find_densities(available_artefacts)
                probabilities = densities / np.sum(densities)

            # if based on frecency, the populations doesn't linger on the past.
            # it favors recent and frequent creations.
            if args.interaction_mode == "frecency":
                artefact_ids = available_artefacts[:, 0]
                counts = recommender.get_frecency_counts(artefact_ids)
                probabilities = counts / np.sum(counts)

            choices = np.random.choice(available_artefacts, size=args.n_artefacts,
                                       replace=False, p=probabilities)

            selected_ids += choices[:, 0]
            selected_artefacts = choices[:, 1]

            # make the choices one hot gaain
            selected[i] = one_hot(selected_artefacts)

        # -------------------------------------------------------------------------------------
        # -- ON EPOCH END UPDATES

        # -- with the new artefacts generate a new ball tree
        if args.interaction_mode == "density":
            recommender.generate_ball_tree()

        if args.interaction_mode == 'frecency':
            recommender.update_frecency(selected_ids)

        # -------------------------------------------------------------------------------------
        # -- EXPORT DOMAIN

        if epoch % 25 == 0:
            data["domain"] += [recommender.export()]

        epoch_log_time = f"{time.time() - epoch_start:0.3f}s".ljust(10, " ")
        print(epoch_log_time, end=("\r" if epoch < args.epochs - 1 else "\n"))

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time elapsed: {elapsed:0.3f}s")

    data["timings"] = pd.Series({"time_elapsed": elapsed,
                                 "start_time": start_time,
                                 "end_time": end_time})

    data = {k: v if is_pd(v) else pd.DataFrame(v) for k, v in results.items()}
    return data


def init(args=None):

    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument("-a", "--agents", type=int, default=16, dest="n_agents",
                        help="The number of agents in the simulation. Default: 16")
    parser.add_argument("-e", "--epochs", type=int, default=250, dest="epochs",
                        help="The number of rounds for the simulation. Default: 250")

    parser.add_argument("-rs", "--r_sample_size", type=int, default=20000, dest="sample_size",
                        help="The number of rounds for the simulation. Default: 250")
    parser.add_argument("-as", "--a_sample_size", type=int, default=3000,
                        dest="agent_sample_size",
                        help="The number of rounds for the simulation. Default: 250")

    parser.add_argument("-f", "--force", action='store_true',
                        help="Forces  the generation of new seeds and models, otherwise \
                              continues with existing seeds and only train models that  \
                              do not exist yet.")

    args = parser.parse_args()
    print(args)

    # -----------------------------------------------------------------------------------------
    # -- DOMAIN

    domain_seed_path = os.path.join(data_path, "seeds", f"domain_seed.npy")

    if os.path.exists(domain_seed_path):
        domain_seed = np.load(domain_seed_path)
    else:
        domain_seed = generate_midi_data(args.sample_size, TIMESTEPS, midi_range=MIDI_RANGE)
        np.save(domain_seed_path, domain_seed)

    recommender = Recommender(domain_seed)

    # -----------------------------------------------------------------------------------------
    # -- INDIVIDUALS

    agent_seed_path = os.path.join(data_path, "seeds", f"agent_seeds.npy")

    if os.path.exists(agent_seed_path):
        agent_seeds = np.load(agent_seed_path)
    else:
        agent_seeds = recommender.generate_seeds(args.n_agents, args.agent_sample_size)
        np.save(agent_seed_path, agent_seeds)

    agents = [Agent(i, seed) for i, seed in enumerate(agent_seeds)]

    return 0


def main(args=None):
    parser = argparse.ArgumentParser()

    # -- basic parameters
    parser.add_argument("-a", "--agents", type=int, default=16, dest="n_agents",
                        help="The number of agents in the simulation. Default: 16")
    parser.add_argument("-e", "--epochs", type=int, default=250, dest="epochs",
                        help="The number of rounds for the simulation. Default: 250")
    parser.add_argument("-n", "--neighbours", type=int, default=2, dest="n_neighbours",
                        help="The number of agents selected to be the field. Default: 2")
    parser.add_argument("-s", "--artefacts", type=int, default=10, dest="n_artefacts",
                        help="The number of artefacts selected each round and the number of \
                              starting artefacts. Default: 10")

    # -- individual paramaters and modes
    parser.add_argument("-sm", "--sample-mode", type=str, default="sample", dest="sample_mode",
                        help="Sets the sample mode for the individual. Options: \
                             'mean', 'sample'")
    parser.add_argument("-nm", "--novelty-mode", type=str, default="density",
                        dest="novelty_mode",
                        help="Sets the novelty mode for the individual. Options: \
                             'density', 'distance'")
    parser.add_argument("--radius", type=float, default=1.0, dest="novelty",
                        help="Sets the nearest-neighbour radius for the agents. Default: 1.0")

    # -- field parameters and modes
    parser.add_argument("-im", "--interaction-mode", type=str, default="uniform",
                        dest="interaction_mode",
                        help="Sets the interaction mode for the field. Options: \
                              'uniform', 'frequency', 'density'")

    # -- export parameters
    parser.add_argument("--save-remote", action="store_true"
                        help="If set to true store results on Amazon S3. Make sure that the \
                              credentials are provided (~/.aws/credentials).")
    args = parser.parse_args()

    # -----------------------------------------------------------------------------------------
    # -- RUN

    t = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime())
    results = run_simulation(args)

    # -----------------------------------------------------------------------------------------
    # -- SAVE

    file_name = f"sim_a{args.n_agents}_e{args.epochs}_{t}"

    if save_remote:
        print("Uploading...", end=" ")
        upload_obj(result, "state-of-the-artefact", f"{file_name}.gz", compression="gzip")
    else:
        print("Saving...", end=" ")
        series = pd.Series(results)
        series.to_pickle(os.path.join(data_path, "output", f"{file_name}.gzip"))

    print("Done.")
    return 0


=======
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


>>>>>>> ed6bc30df6965ef02a03342dec8910f7aaea45cd
if __name__ == "__main__":
    main()
