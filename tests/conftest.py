import pytest

from mml.populations.lens_populations import LensParams
from mml.populations.source_populations import SourceParams
from mml.populations.positions import BBHPositionSample


@pytest.fixture
def lens():
    return LensParams(
        sigma_lens=220.0,
        z_lens=0.5,
        q_lens=0.8,
        ell_mass_lens=0.2,
        theta_mass_lens=0.0,
        ell_light_lens=0.2,
        theta_light_lens=0.0,
        mag_lens=20.0,
        re_lens=1.0,
        x_lens=0.0,
        y_lens=0.0,
        e1_lens=0.1,
        e2_lens=0.0,
        gamma=2.0,
        gamma1=0.0,
        gamma2=0.0,
        theta_ein=0.5,
    )


@pytest.fixture
def source():
    return SourceParams(
        m_VIS_Euclid=22.0,
        log10_mStar=10.0,
        Re_maj_source=0.05,
        z_source=1.5,
        q_source=0.8,
        n_sersic_source=1.0,
        log_p_source=0.0,
        theta_light_source=0.0,
        e1_source=0.1,
        e2_source=0.0,
    )


@pytest.fixture
def position():
    return BBHPositionSample(
        x_gw=0.1,
        y_gw=0.1,
        caustic_area=1.0,
        x_source=0.1,
        y_source=0.1,
    )