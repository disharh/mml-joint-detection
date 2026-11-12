"""
Source-plane regions in which a point source produces a given number of
(bright) lensed images, for an EPL + external-shear lens.

The three regions are *nested* "at least k images" regions::

    quad (>=4)  ⊂  triple (>=3)  ⊂  double (>=2)

* ``quad``   : inside the diamond caustic (and inside the cut where it exists).
* ``triple`` : inside the diamond caustic only (the 4-image region plus the
               "naked-cusp" 3-image sliver outside the radial caustic).
* ``double`` : inside the union of the diamond caustic and the cut / finite
               magnification curve (|mu| = 0.1 for gamma > 2). Contains the
               triple and quad regions.

A source sampled uniformly in ``region(k)`` therefore produces *at least* k
images, not exactly k. How many of them are *detectable* is a separate
question, answered later by the detectors.
"""

from lenstronomy.LensModel.Solver.epl_shear_solver import caustics_epl_shear

# |mu|^-1 cut used for the outer curve of the 'double' region. Do not sample
# images fainter than |mu| = 0.1.
MAGINF_CUT = -1 / 0.1

REGION_NAMES = {2: "double", 3: "triple", 4: "quad"}
_NAME_TO_MIN_IMAGES = {name: n for n, name in REGION_NAMES.items()}


def region_min_images(region):
    """Return the minimum image multiplicity (2, 3 or 4) of a region.

    Parameters
    ----------
    region : int or str
        2/3/4 or "double"/"triple"/"quad".
    """
    if isinstance(region, str):
        try:
            return _NAME_TO_MIN_IMAGES[region.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown region '{region}'. "
                f"Choose from {sorted(_NAME_TO_MIN_IMAGES)}."
            ) from None
    if region not in REGION_NAMES:
        raise ValueError(
            f"Unsupported number of lensed images: {region}. "
            f"Choose from {sorted(REGION_NAMES)}."
        )
    return int(region)


def region_name(region):
    """Return "double", "triple" or "quad" for 2, 3, 4 (or the name itself)."""
    return REGION_NAMES[region_min_images(region)]


def get_caustics(kwargs_lens, number_images):
    """
    Return the source-plane boundary of the region producing *at least*
    ``number_images`` lensed images (2, 3 or 4) for an EPL+shear lens.

    Parameters
    ----------
    kwargs_lens : list of dict
        Lens parameters in lenstronomy format for ['EPL', 'SHEAR'].
    number_images : int or str
        4 / "quad", 3 / "triple", 2 / "double".

    Returns
    -------
    ndarray, shape (2, N)
        Boundary coordinates (x, y) in the source plane.
    """
    n = region_min_images(number_images)

    if n == 4:
        return caustics_epl_shear(kwargs_lens, return_which="quad")
    if n == 3:
        # Diamond caustic. The faint central image (|mu| < 0.01) is ignored.
        return caustics_epl_shear(kwargs_lens, return_which="caustic")
    return caustics_epl_shear(
        kwargs_lens, return_which="double", maginf=MAGINF_CUT
    )