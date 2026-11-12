import pytest

from mml.simulation.sampling import resolve_sampling


@pytest.mark.parametrize(
    "k, region", [(2, 2), (3, 3), (4, 4), (5, 4)]
)
def test_auto_picks_smallest_complete_region(k, region):
    assert resolve_sampling(k, "auto") == (region, True)


def test_larger_region_than_needed_is_still_complete():
    # >=2 detectable images sampled from the (larger) double region
    assert resolve_sampling(4, "double") == (2, True)
    assert resolve_sampling(3, "double") == (2, True)


def test_smaller_region_than_needed_is_a_subpopulation():
    # quad lenses only, requiring >=2 detectable: not the full population
    assert resolve_sampling(2, "quad") == (4, False)
    assert resolve_sampling(2, "triple") == (3, False)


def test_rejects_single_image_requirement():
    with pytest.raises(ValueError):
        resolve_sampling(1)