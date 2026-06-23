import argparse

from .resolvit import (
    process_observation,
    DEFAULT_BIN_SIZE,
    DEFAULT_TOTAL_EVENTS_FRACTION,
    __version__,
)


def main():

    parser = argparse.ArgumentParser(
        prog="resolvit",
        description=(
            "Improve the PSF of UVIT Level2 event lists and "
            "generate Resolvit data products."
        ),
    )

    parser.add_argument(
        "observation_dir",
        help="Path to UVIT Level2 observation directory",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "--bin-size",
        type=float,
        default=DEFAULT_BIN_SIZE,
        metavar="SECONDS",
        help=(f"Time bin size in seconds (default: {DEFAULT_BIN_SIZE})"),
    )

    parser.add_argument(
        "--event-fraction",
        type=float,
        default=DEFAULT_TOTAL_EVENTS_FRACTION,
        metavar="FRACTION",
        help=(
            f"Fraction of median events required for a bin (default: {DEFAULT_TOTAL_EVENTS_FRACTION})"
        ),
    )

    args = parser.parse_args()

    process_observation(
        args.observation_dir,
        bin_size=args.bin_size,
        total_events_fraction=args.event_fraction,
    )


if __name__ == "__main__":
    main()
