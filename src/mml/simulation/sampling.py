"""Choice of the source-plane sampling region for a simulation run."""

from mml.lensing.caustics import region_min_images


def resolve_sampling(min_detected: int, sampling_region="auto"):
    """
    Decide which source-plane region to sample BBH positions from.

    Parameters
    ----------
    min_detected : int
        Required number of detectable lensed images (>= 2).
    sampling_region : {"auto", "double", "triple", "quad", 2, 3, 4}
        "auto" picks the smallest region that still contains every system
        with ``min_detected`` detectable images.

    Returns
    -------
    region_min : int
        2, 3 or 4: minimum number of images produced inside the region.
    complete : bool
        True if the region contains every system with >= ``min_detected``
        detectable images (so ``caustic_area`` is the correct cross-section
        for the "at least K detectable images" population).
    """
    if min_detected < 2:
        raise ValueError("min_detected must be >= 2.")

    needed = min(min_detected, 4)

    if sampling_region in ("auto", None):
        region_min = needed
    else:
        region_min = region_min_images(sampling_region)

    # A region produces at least `region_min` images, so it contains all
    # systems with >= min_detected detectable images iff region_min <= needed.
    return region_min, region_min <= needed