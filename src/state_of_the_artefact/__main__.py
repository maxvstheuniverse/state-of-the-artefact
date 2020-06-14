import argparse


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=4, dest="agents",
                        help="The number of agents in the simulation. Default: 4")
    args = parser.parse_args()

    print("Hello Universe!")
    return 0
