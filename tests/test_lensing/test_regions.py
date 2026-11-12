"""The source-plane regions are nested 'at least k bright images' regions."""

import numpy as np
import pytest
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from mml.lensing import get_caustics, sample_polygon_single
from mml.lensing.caustics import region_min_images, region_name


def make_lens(gamma):
    return [
        {"theta_E": 0.8, "gamma": gamma, "e1": -0.12, "e2": -0.2,
         "center_x": 0.0, "center_y": 0.0},
        {"gamma1": -0.04, "gamma2": 0.07},
    ]


# gamma != 2 on purpose: gamma == 2 exactly is a degenerate case for the
# 'double' curve and is measure-zero for a continuous slope distribution.
LENSES = [make_lens(1.85), make_lens(2.12)]


def polygon_area(poly):
    x, y = poly
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def n_bright_images(kwargs_lens, x, y, mu_min=0.1):
    model = LensModel(["EPL_NUMBA", "SHEAR"])
    solver = LensEquationSolver(model)
    xi, yi = solver.image_position_from_source(
        x, y, kwargs_lens, solver="analytical"
    )
    mu = np.abs(model.magnification(xi, yi, kwargs_lens))
    return int(np.sum(mu >= mu_min))


def test_region_helpers():
    assert region_min_images("quad") == 4
    assert region_min_images(3) == 3
    assert region_name(2) == "double"
    assert region_name("triple") == "triple"
    for bad in (1, 5, "penta"):
        with pytest.raises(ValueError):
            region_min_images(bad)


@pytest.mark.parametrize("kwargs_lens", LENSES)
def test_regions_are_nested_by_area(kwargs_lens):
    a2, a3, a4 = (
        polygon_area(get_caustics(kwargs_lens, n)) for n in (2, 3, 4)
    )
    assert a4 <= a3 * (1 + 1e-6) <= a2 * (1 + 1e-6)


@pytest.mark.parametrize("n_min", [2, 3, 4])
@pytest.mark.parametrize("kwargs_lens", LENSES)
def test_sampling_in_region_gives_at_least_n_bright_images(kwargs_lens, n_min):
    rng = np.random.default_rng(0)
    poly = get_caustics(kwargs_lens, n_min)

    counts = []
    for _ in range(150):
        (x, y), _ = sample_polygon_single(poly, rng.uniform(size=2))
        counts.append(n_bright_images(kwargs_lens, x, y))

    # >= n_min bright images (allow a sliver of boundary points)
    assert np.mean(np.array(counts) >= n_min) > 0.97


def test_double_region_also_contains_quads():
    """'double' means >=2, not exactly 2: it must include 4-image sources."""
    kwargs_lens = LENSES[1]
    rng = np.random.default_rng(1)
    poly = get_caustics(kwargs_lens, 2)

    counts = set()
    for _ in range(400):
        (x, y), _ = sample_polygon_single(poly, rng.uniform(size=2))
        counts.add(n_bright_images(kwargs_lens, x, y))

    assert 2 in counts and 4 in counts