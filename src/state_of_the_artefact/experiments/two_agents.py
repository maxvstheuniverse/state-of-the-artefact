import argparse
import numpy as np
import os
import pandas as pd
import random
import shortid
import time

from state_of_the_artefact.systems import Agent
from state_of_the_artefact.representation import generate_midi_data_tonic, create_ctable

TIMESTEPS = 16
BATCH_SIZE = 32
MIDI_RANGE = range(24, 36)

sid = shortid.ShortId()


def two_agents(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=100, dest="epochs",
                        help="The number of rounds for the simulation. Default: 100")
    parser.add_argument("-ss", "--sample-size", type=int, default=1000, dest="sample_size",
                        help="The number of samples in the synthetic dataset. Default: 1000")
    parser.add_argument("-i", "--init-epochs", type=int, default=250, dest="init_epochs",
                        help="The number of epochs used for schooling the individuals. Default: 250")
    parser.add_argument("-s", "--artefacts", type=int, default=5, dest="n_artefacts",
                        help="The number of item selected each round. Default: 5")
    args = parser.parse_args()

    # generate data path
    seed_path = os.path.join(os.getcwd(), "data", "seeds", "two_agents")
    data_path = os.path.join(os.getcwd(), "data", "output", "two_agents")
    model_path = os.path.join(os.getcwd(), "data", "models", "two_agents")

    if not os.path.exists(seed_path):
        os.makedirs(seed_path)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # setup the representation
    characters = [f"{pitch}" for pitch in MIDI_RANGE]
    encode, decode, vectorize = create_ctable(characters)

    load_seed = False  # temp fix.
    if load_seed:
        seed_one = np.load(os.path.join(seed_path, f"seed_one_C_{args.sample_size}.npy"))
        seed_two = np.load(os.path.join(seed_path, f"seed_two_C#_{args.sample_size}.npy"))
    else:
        # generate two seed of different types, of the given samples size
        seed_one = vectorize(generate_midi_data_tonic(args.sample_size, TIMESTEPS, key="C"))
        seed_two = vectorize(generate_midi_data_tonic(args.sample_size, TIMESTEPS, key="F"))

        np.save(os.path.join(seed_path, f"seed_one_C_{args.sample_size}.npy"), seed_one)
        np.save(os.path.join(seed_path, f"seed_two_C#_{args.sample_size}.npy"), seed_two)

    both_seeds = np.concatenate([seed_one, seed_two])

    # make two agents
    agent_one = Agent(1, "C")
    agent_two = Agent(2, "C#")

    # fit the agents, for the deterimed initial number of epochs
    agent_one.fit(seed_one, epochs=args.init_epochs, model_path=model_path, batch_size=BATCH_SIZE)
    agent_two.fit(seed_two, epochs=args.init_epochs, model_path=model_path, batch_size=BATCH_SIZE)

    metrics = {"evaluations": [], "reconstructions": []}

    initial_evaluation = {
        "one_native": agent_one.evaluate(seed_one),
        "two_native": agent_two.evaluate(seed_two),
        "one_other": agent_one.evaluate(seed_two),
        "two_other": agent_two.evaluate(seed_one)
    }
    metrics["evaluations"].append(initial_evaluation)

    # set start n artefacts
    selected = [np.array(random.choices(seed_one, k=5)), np.array(random.choices(seed_two, k=5))]

    start_time = time.time()
    t = time.strftime('%Y-%m-%dT%H-%M-%S', time.localtime())

    # start simulation loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch:04d}...", end=" ")

        for i, agent in enumerate([agent_one, agent_two]):
            # setup on first epoch
            if epoch == 0:
                z_means, _, _ = agent.encode(selected[i])

                for artefact, z_mean in zip(selected[i], z_means):
                    entry = [-1, agent.id, agent.culture_id, sid.generate(), artefact, z_mean]
                    agent.store(entry)

            # learn
            ideal = agent.learn(selected[i], apply_mean=True)

            # build
            new_artefact, z_mean = agent.build(ideal)

            # store
            entry = [epoch, agent.id, agent.culture_id, sid.generate(), new_artefact[0], z_mean]
            agent.store(entry)

        # evaluation of the model
        evaluation = {
            "one_native": agent_one.evaluate(seed_one),
            "two_native": agent_two.evaluate(seed_two),
            "one_other": agent_one.evaluate(seed_two),
            "two_other": agent_two.evaluate(seed_one)
        }
        metrics["evaluations"].append(evaluation)

        # reconstruct everything
        reconstruction = {
            "agent_one": agent_one.reconstruct(),
            "agent_two": agent_two.reconstruct()
        }
        metrics["reconstructions"].append(reconstruction)

        # select and share
        artefacts = agent_one.select() + agent_two.select()
        selected = [np.array(random.choices(artefacts, k=5)), np.array(random.choices(artefacts, k=5))]

        print(f"{time.time() - epoch_start:0.3f}s", end=("\r" if epoch < args.epochs - 1 else "\n"))

    print(f"Total time elapsed: {time.time() - start_time:0.3f}s")
    print("Saving...", end=" ")

    # Convert metrics to DataFrames and save results to pickle
    data = pd.Series({k: pd.DataFrame(v) for k, v in metrics.items()})
    data.to_pickle(os.path.join(data_path, f"{args.init_epochs}_{args.sample_size}_{t}.gzip"))

    print("Done.")
    return 0