import numpy as np
import pytest

from mml.populations.gw_populations import (
    GWParams,
    LensedGWParams,
    GWPopulation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gw():
    """Return a minimal GWParams object for testing."""

    return GWParams(
        mass_1_source=30.0,
        mass_2_source=20.0,
        a_1=0.5,
        a_2=0.3,
        tilt_1=0.4,
        tilt_2=0.6,
        phi_12=1.0,
        phi_jl=2.0,
        theta_jn=1.2,
        ra=1.0,
        dec=0.5,
        psi=0.7,
        geocent_time=100.0,
        phase=0.2,
        luminosity_distance=1000.0,
    )


# ---------------------------------------------------------------------------
# GWParams
# ---------------------------------------------------------------------------

def test_gwparams_to_dict():
    gw = make_gw()

    params = gw.to_dict()

    assert isinstance(params, dict)
    assert params["mass_1_source"] == 30.0
    assert params["mass_2_source"] == 20.0
    assert params["luminosity_distance"] == 1000.0


# ---------------------------------------------------------------------------
# Morse phase
# ---------------------------------------------------------------------------

def test_morse_phase_minimum_cases():
    hessian = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    phase = GWPopulation._morse_phase(hessian)

    assert phase.shape == (3,)
    assert np.allclose(phase, 0.0)


def test_morse_phase_saddle():
    # det(A) < 0
    #
    # A = [[2, 0],
    #      [0, -1]]
    #
    # This corresponds to
    # f_xx = -1, f_yy = 2.
    hessian = np.array([
        [-1.0],
        [0.0],
        [0.0],
        [2.0],
    ])

    phase = GWPopulation._morse_phase(hessian)

    assert np.allclose(phase, np.pi / 2)


def test_morse_phase_maximum():
    # det(A) > 0 and trace(A) < 0
    #
    # A = [[-1, 0],
    #      [0, -1]]
    hessian = np.array([
        [2.0],
        [0.0],
        [0.0],
        [2.0],
    ])

    phase = GWPopulation._morse_phase(hessian)

    assert np.allclose(phase, np.pi)


# ---------------------------------------------------------------------------
# Lensing
# ---------------------------------------------------------------------------

def test_lens_returns_lensed_gw(monkeypatch):
    gw = make_gw()

    class FakeLensModel:
        def __init__(self, *args, **kwargs):
            pass

        def magnification(self, x, y, kwargs):
            return np.array([4.0, 1.0])

        def arrival_time(self, x_image, y_image, kwargs_lens):
            return np.array([1.0, 3.0])

        def hessian(self, x, y, kwargs):
            return np.array([
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ])

    class FakeSolver:
        def __init__(self, lens_model):
            pass

        def image_position_from_source(
            self,
            x,
            y,
            kwargs_lens,
            solver,
        ):
            return (
                np.array([0.1, -0.1]),
                np.array([0.2, -0.2]),
            )

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensModel",
        FakeLensModel,
    )

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensEquationSolver",
        FakeSolver,
    )

    class FakeLens:
        z_lens = 0.5

        def to_lenstronomy(self):
            return [
                {"theta_ein": 1.0},
                {"gamma1": 0.0, "gamma2": 0.0},
            ]

    population = GWPopulation()

    result = population.lens(
        gw=gw,
        lens=FakeLens(),
        x_gw=0.0,
        y_gw=0.0,
        z_source=1.0,
    )

    assert isinstance(result, LensedGWParams)
    assert result.n_images == 2

    assert result.x_image.shape == (2,)
    assert result.y_image.shape == (2,)
    assert result.magnifications.shape == (2,)
    assert result.time_delays.shape == (2,)

    assert result.effective_luminosity_distance.shape == (2,)
    assert result.effective_geocent_time.shape == (2,)
    assert result.effective_phase.shape == (2,)
    assert result.effective_ra.shape == (2,)
    assert result.effective_dec.shape == (2,)


def test_lens_distance_scaling(monkeypatch):
    """
    Check dL_eff = dL / sqrt(|mu|).
    """

    gw = make_gw()

    class FakeLensModel:
        def __init__(self, *args, **kwargs):
            pass

        def magnification(self, x, y, kwargs):
            return np.array([4.0, 1.0])

        def arrival_time(self, x_image, y_image, kwargs_lens):
            return np.array([0.0, 1.0])

        def hessian(self, x, y, kwargs):
            return np.zeros((4, 2))

    class FakeSolver:
        def __init__(self, lens_model):
            pass

        def image_position_from_source(
            self,
            x,
            y,
            kwargs_lens,
            solver,
        ):
            return (
                np.array([0.1, 0.2]),
                np.array([0.1, 0.2]),
            )

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensModel",
        FakeLensModel,
    )

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensEquationSolver",
        FakeSolver,
    )

    class FakeLens:
        z_lens = 0.5

        def to_lenstronomy(self):
            return [{}, {}]

    result = GWPopulation().lens(
        gw=gw,
        lens=FakeLens(),
        x_gw=0.0,
        y_gw=0.0,
        z_source=1.0,
    )

    expected = np.array([
        1000.0 / np.sqrt(4.0),
        1000.0 / np.sqrt(1.0),
    ])

    assert np.allclose(
        result.effective_luminosity_distance,
        expected,
    )


def test_lens_time_delays_are_relative(monkeypatch):
    """The earliest image should have zero time delay."""

    gw = make_gw()

    class FakeLensModel:
        def __init__(self, *args, **kwargs):
            pass

        def magnification(self, x, y, kwargs):
            return np.ones(2)

        def arrival_time(self, x_image, y_image, kwargs_lens):
            return np.array([5.0, 2.0])

        def hessian(self, x, y, kwargs):
            return np.zeros((4, 2))

    class FakeSolver:
        def __init__(self, lens_model):
            pass

        def image_position_from_source(
            self,
            x,
            y,
            kwargs_lens,
            solver,
        ):
            return (
                np.array([0.1, 0.2]),
                np.array([0.1, 0.2]),
            )

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensModel",
        FakeLensModel,
    )

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensEquationSolver",
        FakeSolver,
    )

    class FakeLens:
        z_lens = 0.5

        def to_lenstronomy(self):
            return [{}, {}]

    result = GWPopulation().lens(
        gw=gw,
        lens=FakeLens(),
        x_gw=0.0,
        y_gw=0.0,
        z_source=1.0,
    )

    assert np.allclose(
        result.time_delays,
        np.array([3.0, 0.0]) * 86400.0,
    )


def test_lens_rejects_zero_images(monkeypatch):
    gw = make_gw()

    class FakeLensModel:
        def __init__(self, *args, **kwargs):
            pass

    class FakeSolver:
        def __init__(self, lens_model):
            pass

        def image_position_from_source(
            self,
            x,
            y,
            kwargs_lens,
            solver,
        ):
            return np.array([]), np.array([])

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensModel",
        FakeLensModel,
    )

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensEquationSolver",
        FakeSolver,
    )

    class FakeLens:
        z_lens = 0.5

        def to_lenstronomy(self):
            return [{}, {}]

    with pytest.raises(
        RuntimeError,
        match="no images",
    ):
        GWPopulation().lens(
            gw=gw,
            lens=FakeLens(),
            x_gw=0.0,
            y_gw=0.0,
            z_source=1.0,
        )


def test_lens_rejects_more_than_five_images(monkeypatch):
    gw = make_gw()

    class FakeLensModel:
        def __init__(self, *args, **kwargs):
            pass

    class FakeSolver:
        def __init__(self, lens_model):
            pass

        def image_position_from_source(
            self,
            x,
            y,
            kwargs_lens,
            solver,
        ):
            return np.arange(6), np.arange(6)

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensModel",
        FakeLensModel,
    )

    monkeypatch.setattr(
        "mml.populations.gw_populations.LensEquationSolver",
        FakeSolver,
    )

    class FakeLens:
        z_lens = 0.5

        def to_lenstronomy(self):
            return [{}, {}]

    with pytest.raises(
        RuntimeError,
        match="6 images",
    ):
        GWPopulation().lens(
            gw=gw,
            lens=FakeLens(),
            x_gw=0.0,
            y_gw=0.0,
            z_source=1.0,
        )