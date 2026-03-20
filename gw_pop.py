import ler
from ler.gw_source_population import CBCSourceParameterDistribution
import numpy as np

def sample_gw_params(size=1):
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


def compute_morse_phase(hessian):
    """
    Computes morse phase per image using hessian.
    """
    f_xx, f_xy, f_yx, f_yy = hessian

    A_xx = 1 - f_xx
    A_xy = -f_xy
    A_yx = -f_yx
    A_yy = 1 - f_yy

    detA = A_xx * A_yy - A_xy * A_yx
    traceA = A_xx + A_yy

    morse = np.zeros_like(detA)

    morse[(detA < 0)] = np.pi / 2.0          # saddle
    morse[(detA > 0) & (traceA < 0)] = np.pi  # maximum
    morse[(detA > 0) & (traceA > 0)] = 0.0    # minimum

    return morse

def compute_effective_gw_params(
    gw_params,
    magnifications,
    delays,
    hessian,
    x_image,
    y_image,
    x_gw,
    y_gw,
):
    """
    Returns effective GW parameters per image.

    Parameters
    ----------
    gw_params : Dictionary of gw_params 
    magnifications : Output of lenstronomy.LensModel.magnification
    delays : Time delay array in units of seconds
    hessian : Output of lenstronomy.LensModel.hessian
    x_image, y_image : Image positions in angular coordinates (arcsecs)
    x_gw, y_gw : Angular GW position coordinates (arcsecs)

    """
    morse_phase = compute_morse_phase(hessian)

    dL = gw_params["luminosity_distance"][:, np.newaxis]  
    t0 = gw_params["geocent_time"][:, np.newaxis]
    phi = gw_params["phase"][:, np.newaxis]
    ra = gw_params["ra"][:, np.newaxis]
    dec = gw_params["dec"][:, np.newaxis]

    # lensing quantities
    mu = np.abs(magnifications)[np.newaxis, :] 
    dt = delays[np.newaxis, :]                
    morse = morse_phase[np.newaxis, :]         

    # positions
    arcsec_to_rad = 1.0 / 206265.0
    dx = (x_image[np.newaxis, :] - x_gw) * arcsec_to_rad
    dy = (y_image[np.newaxis, :] - y_gw) * arcsec_to_rad      
    cosdec = np.cos(dec)  

    # effective parameters 
    gw_params["effective_luminosity_distance"] = dL / np.sqrt(mu)
    gw_params["effective_geocent_time"] = t0 + dt
    gw_params["effective_phase"] = phi - morse
    gw_params["effective_ra"] = ra + dx / cosdec
    gw_params["effective_dec"] = dec + dy

    return gw_params