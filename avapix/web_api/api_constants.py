import os


def get_full_path(relative_path: str) -> str:
    """
    Get the full path of a file based on a relative path.

    Parameters
    ----------
    relative_path : str
        Relative path of the file.

    Returns
    -------
    str
        Full path of the file.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


AVATAR_DIR = get_full_path("static")
