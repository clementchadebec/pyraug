'''https://www.doc.ic.ac.uk/~nuric/coding/argparse-with-multiple-files-to-handle-configuration-in-python.html'''

from typing import Dict, Any
import logging
import pprint
import sys
import argparse

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")

config: Dict[str, Any] = {}


def add_parser(title: str, description: str = ""):
    """Create a new context for arguments and return a handle."""
    return parser.add_argument_group(title, description)


def parse(save_fname: str = "") -> Dict[str, Any]:
    """Parse given arguments."""
    config.update(vars(parser.parse_args()))
    logging.info("Parsed %i arguments.", len(config))
    # Save passed arguments
    if save_fname:
        with open(save_fname, "w") as fout:
            fout.write("\n".join(sys.argv[1:]))
        logging.info("Saving arguments to %s.", save_fname)
    return config


def print_config():
    """Print the current config to stdout."""
    pprint.pprint(config)