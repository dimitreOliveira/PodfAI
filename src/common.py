import json
import logging

import yaml

ACCEPTED_FILE_INPUTS = [
    "txt",
    "md",
    "pdf",
]


def parse_configs(configs_path: str) -> dict:
    """Parse configs from the YAML file.

    Args:
        configs_path (str): Path to the YAML file

    Returns:
        dict: Parsed configs
    """
    configs = yaml.safe_load(open(configs_path, "r"))
    logger.info(f"Configs: \n{json.dumps(configs, indent=4)}")
    return configs


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
