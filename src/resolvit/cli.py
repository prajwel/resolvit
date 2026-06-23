import argparse

from .resolvit import process_observation


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "observation_dir",
        help="Path to UVIT Level2 observation directory",
    )

    args = parser.parse_args()

    process_observation(
        args.observation_dir,
    )


if __name__ == "__main__":
    main()
