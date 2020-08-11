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
[x] test s3 on liacs computers
[x] avoid duplicates
[x] frecency

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
SAMPLES = 10000

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

    parameters = pd.Series(vars(args))

    data = {"parameters": parameters,
            "evaluations": [],
            "reconstructions": [],
            "interactions": []}

    # -----------------------------------------------------------------------------------------
    # -- SETUP

    domain_seed_path = os.path.join(data_path, "seeds", "domain_seed.npy")
    agent_seeds_path = os.path.join(data_path, "seeds", "agent_seeds.npy")

    try:
        domain_seed = np.load(domain_seed_path, allow_pickle=True)
        agent_seeds = np.load(agent_seeds_path, allow_pickle=True)
    except IOError as e:
        print(e.args)
        return 1

    recommender = Recommender(domain_seed, args.i_epochs)
    agents = [Agent(i, seed, args.i_epochs) for i, seed in enumerate(agent_seeds)]

    # -- Other sim settings

    uniform_mode = args.interaction_mode == 'uniform'
    frecency_mode = args.interaction_mode == 'frecency'
    density_mode = args.interaction_mode == 'density'


    # -----------------------------------------------------------------------------------------
    # -- EPOCH 0

    # -- prepare initial artefacts
    # seed[np.random.choice(np.arange(len(seed)), size=args.n_select, replace=False)]

    # returns the artefacts and their z_means
    initial_artefacts = [agent.build(agent.sample(1, args.n_select))
                         for agent in agents]

    for agent, (artefacts, z_means) in zip(agents, initial_artefacts):
        initial_entries = [make_entry(0, agent, artefact, z_mean)
                           for artefact, z_mean in zip(artefacts, z_means)]

        recommender.save(initial_entries, frecency=frecency_mode)

    selected = [artefact[0] for artefact in initial_artefacts]
    assert np.array(selected).shape == (args.n_agents, args.n_select, 16, 12)

    if args.interaction_mode == "density":
        # -- generate the first ball tree
        recommender.generate_ball_tree()

    print("\nStarting simulation with the following parameters:")
    print(parameters.to_string(), end="\n\n")

    # -----------------------------------------------------------------------------------------
    # -- LOOP

    for epoch in range(1, args.n_epochs + 1):
        print(f"Epoch {epoch:03d}...", end=" ")
        epoch_start = time.time()

        # -------------------------------------------------------------------------------------
        # -- INDIVIDUALS

        new_entries = []
        evaluation = {}
        reconstruction = {}

        for i, agent in enumerate(agents):
            # learn
            z_means = agent.learn(selected[i], args.budget)

            # sample n new artefacts
            if args.production_mode == "interpolate":
                z = agent.interpolate(z_means)

            if args.production_mode == "origin":
                z = agent.sample(args.novelty_preference, args.n_artefacts)

                # calculate novelty values?

            artefacts, _ = agent.build(z)
            # entry = make_entry(epoch, agent, artefacts[0], z_mean)

            agent_entries = [make_entry(epoch, agent, artefacts[0], 0)
                             for artefact in artefacts]

            # store
            new_entries.append(agent_entries)

        # -------------------------------------------------------------------------------------
        # -- DOMAIN

        positions = recommender.find_positions(new_entries,
                                               save_entries=True,
                                               frecency=frecency_mode)

        # -------------------------------------------------------------------------------------
        # -- FIELD

        selected_ids = []

        # TODO: implement different social interaction policies.
        for i, agent in enumerate(agents):

            # -- FIELD NEIGHBOURS

            # calculate distances to other agents
            distances = np.linalg.norm(positions[i] - positions, axis=1)

            # get neighbours, sort, return indices, remove current agent
            neighbours = [nghbr for nghbr in np.argsort(distances) if nghbr != agent.id]
            assert agent.id not in neighbours, "Current agent is present in neighbour list."

            # record interactions
            data["interactions"] += [{"epoch": epoch,
                                      "agent_id": agent.id,
                                      "position": positions[i],
                                      "neighbours": neighbours,
                                      "distances": distances}]

            # gather artefacts created by the field
            available_artefacts = recommender.select_artefacts(agent.id,
                                                               with_ids=(not uniform_mode))

            for neighbour_id in neighbours:
                neighbour_artefacts = recommender.select_artefacts(neighbour_id,
                                                                   with_ids=(not uniform_mode))
                available_artefacts = np.vstack([available_artefacts, neighbour_artefacts])

            # -- FIELD MODE

            # default is uniform, which means no ideology in the population
            probabilities = None

            if not uniform_mode:
                artefact_ids = available_artefacts[:, 0]
                available_artefacts = np.vstack(available_artefacts[:, 1])

                # if based on density, the population favors artefacts which are "more"
                # unique, based on the estimated density in their space.
                # Calculates the inverse in order to favor low density artefacts.

                if density_mode:
                    # artefact_ids = available_artefacts[:, 0]
                    densities = 1 / recommender.find_densities(artefact_ids)
                    probabilities = densities / np.sum(densities)

                # if based on frecency, the populations doesn't linger on the past.
                # it favors recent and frequent creations.

                if frecency_mod:
                    # artefact_ids = available_artefacts[:, 0]
                    counts = recommender.get_frecency_counts(artefact_ids)
                    probabilities = counts / np.sum(counts)

            # -- FIELD SELECTION
            choices = np.random.choice(np.arange(len(available_artefacts)),
                                       size=args.n_select,
                                       replace=False,
                                       p=probabilities)

            selected_artefacts = available_artefacts[choices]

            if args.interaction_mode == "frecency":
                selected_ids.append(artefact_ids[choices])

            # make the choices one hot gaain
            selected[i] = one_hot(selected_artefacts)

            # -- WRAP UP EVALUATIONS FOR EACH AGENT

            agent_entries = recommender.select_entries(agent.id)
            agent_artefacts = one_hot(np.array([entry["artefact"] for entry in agent_entries]))

            # NOTE: considerable computational resources, consider every n epochs
            # CONSIDER: Evaluate on the seed?

            data["evaluations"] += [{"epoch": epoch,
                                     "agent_id": agent.id,
                                     **agent.evaluate(agent_artefacts)}]

            # NOTE: considerable computational resources, consider every n epochs
            data["reconstructions"] += agent.reconstruct(epoch, agent_entries, agent_artefacts)

        # -------------------------------------------------------------------------------------
        # -- ON EPOCH END UPDATES

        # -- with the new artefacts generate a new ball tree
        if args.interaction_mode == "density":
            recommender.generate_ball_tree()

        if args.interaction_mode == 'frecency':
            recommender.update_frecency(np.hstack(selected_ids))

        epoch_log_time = f"{time.time() - epoch_start:0.3f}s".ljust(15, " ")
        print(epoch_log_time, end=("\r" if epoch < args.n_epochs else "\n"))

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total time elapsed: {elapsed:0.3f}s")

    # -------------------------------------------------------------------------------------
    # -- FINALIZE

    # -- export domain
    data["domain"] = pd.DataFrame(recommender.export())

    # -- sample from agents
    after_sample = []

    for agent in agents:
        z = np.random.normal(0.0, 0.5, size=(100, 32))  # just sampling 100 random samples for visualizing clusters.
        artefacts = np.argmax(agent.rvae.decode(z, apply_onehot=False), axis=-1)
        # print(artefacts)
        after_sample.append([{agent.id: artefact} for artefact in artefacts])

    data["after_sample"] = after_sample

    data["timings"] = pd.Series({"duration": elapsed,
                                 "start_time": time.strftime('%Y-%m-%dT%H-%M-%S',
                                                             time.localtime(start_time)),
                                 "end_time": time.strftime('%Y-%m-%dT%H-%M-%S',
                                                           time.localtime(end_time))})

    return {k: v if is_pd(v) else pd.DataFrame(v) for k, v in data.items()}


def init(args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=16, dest="n_agents",
                        help="The number of agents in the simulation. Default: 16")
    parser.add_argument("-e", "--epochs", type=int, default=250, dest="epochs",
                        help="The number of rounds for the simulation. Default: 250")
    parser.add_argument("-ie", "--init-epochs", type=int, default=250, dest="init_epochs",
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

    agents = [Agent(i, seed, init_epochs=args.i_epochs) for i, seed in enumerate(agent_seeds)]

    return 0


def main(args=None):
    parser = argparse.ArgumentParser()

    # -- basic parameters
    parser.add_argument("-e", "--epochs", type=int, default=250, dest="n_epochs",
                        help="The number of rounds for the simulation. Default: 250")
    parser.add_argument("-ie", "--init-epochs", type=int, default=500, dest="i_epochs",
                        help="The number of rounds for the simulation. Default: 500")
    parser.add_argument("-a", "--agents", type=int, default=8, dest="n_agents",
                        help="The number of agents in the simulation. Default: 8")
    parser.add_argument("-n", "--neighbours", type=int, default=1, dest="n_neighbours",
                        help="The number of agents selected to be the field. Default: 1")
    parser.add_argument("-s", "--select", type=int, default=128, dest="n_select",
                        help="The number of artefacts selected each round and the number of \
                              starting artefacts. Default: 128")
    parser.add_argument("-r", "--artefacts", type=int, default=16, dest="n_artefacts",
                        help="The number of artefacts produced each round and the number of \
                              starting artefacts. Default: 16")

    # -- individual paramaters and modes
    parser.add_argument("-pm", "--production-mode", type=str, default="origin",
                        dest="production_mode",
                        help="Sets the sample mode for the individual, use the origin, or the \
                              mean of current artefacts presented by the field. \
                              Options: 'interpolate', 'origin'")
    parser.add_argument("-np", "--novelty-preference", type=float, default=0.25,
                        dest="novelty_preference",
                        help="Standard deviation used when sampling from the latent space. \
                              Default: .25")
    parser.add_argument("-b", "--budget", type=int, default=25, dest="budget",
                        help="Maximum iterations for learning new artefacts presented by \
                              the field. Default: 25")

    # -- field parameters and modes
    parser.add_argument("-im", "--interaction-mode", type=str, default="uniform",
                        dest="interaction_mode",
                        help="Sets the interaction mode for the field. Options: \
                              'uniform', 'frequency', 'density'")

    # -- export parameters
    parser.add_argument("--save-remote", action="store_true",
                        help="If set to true store results on Amazon S3. Make sure that the \
                              credentials are provided (~/.aws/credentials).")
    args = parser.parse_args()

    # -----------------------------------------------------------------------------------------
    # -- RUN

    t = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime())
    results = run_simulation(args)

    # -----------------------------------------------------------------------------------------
    # -- SAVE

    im, sm, np = args.interaction_mode, args.production_mode, args.novelty_preference,
    b, s, r = args.budget, args.n_select, args.n_artefacts

    file_name = f"sim_a{args.n_agents}_e{args.n_epochs}_{im}_{sm}_np{np}_b{b}_sr{s}-{r}_{t}"

    if args.save_remote:
        print("Uploading...", end=" ")
        upload_obj(results, "state-of-the-artefact", f"{file_name}.gz", compression="gzip")
    else:
        print("Saving...", end=" ")
        pd.Series(results).to_pickle(os.path.join(data_path, "output", f"{file_name}.gzip"))

    print("Done.")
    return 0


if __name__ == "__main__":
    main()
