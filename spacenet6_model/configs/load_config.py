import argparse

from .defaults import get_default_config


def load_config():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)

    config.freeze()

    return config
