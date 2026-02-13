import ler
from ler.gw_source_population import CBCSourceParameterDistribution

def sample_gw_params(size):
    """
    Sample BBH GW parameters (intrinsic + extrinsic) with precessing spins using LeR.

    Parameters
    ----------
    size : number of samples 

    Returns
    ---------- 
    Dictionary of the following sampled params:
    - mass_1_source, mass_2_source
    - a_1, a_2
    - tilt_1, tilt_2
    - phi_12, phi_jl
    - theta_jn
    - ra, dec
    - psi
    - geocent_time
    - phase

    Does not sample redshift, luminosity distance and detector frame masses.
    """

    cbc = CBCSourceParameterDistribution(
        event_type="BBH",
        spin_zero=False,
        spin_precession=True,
    )

    params = cbc.sample_gw_parameters(size=size)

    selected_keys = [
        "mass_1_source",
        "mass_2_source",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "theta_jn",
        "ra",
        "dec",
        "psi",
        "geocent_time",
        "phase",
    ]

    return {key: params[key] for key in selected_keys}