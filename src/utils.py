import logging

import vertexai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_vertex(project: str, location: str) -> None:
    """Setups the Vertex AI project.

    Args:
        project (str): Vertex AI project name
        location (str): Vertex AI project location
    """
    logger.info("Initializing Vertex AI setup")
    vertexai.init(project=project, location=location)
