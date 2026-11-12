from pathlib import Path


def get_cache_dir(name="mml"):
    """
    Return the cache directory used by the package.

    Parameters
    ----------
    name : str
        Name of the subdirectory inside the user cache.

    Returns
    -------
    pathlib.Path
        Cache directory.
    """

    cache_dir = Path.home() / ".cache" / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir