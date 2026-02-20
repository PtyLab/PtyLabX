# all the settings involving configuration go here
from pathlib import Path


def get_PtyLabX_folder():
    """Return the folder that PtyLabX is installed in."""
    return Path(__file__).parent.parent
