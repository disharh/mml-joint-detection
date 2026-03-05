## Script for defining detection and lensing probabilities
# some functions are adapted from Ewoud's code

from image_snr_euclid import *
import numpy as np
from astropy import units as u
from ler.image_properties import ImageProperties
from gwsnr import GWSNR
from lenstronomy.LensModel.lens_model import LensModel

def lik_cross_sec(area):
    """
    Compute the log probability that a random GW occurs within a lens caustic of a given area in the source plane.

    Parameters
    ----------
    area : float
        Area of the caustic in the source plane, in units of arcsec².

    Returns
    -------
    prob : float
        Log probability that a random GW falls inside the caustic area,
        assuming a uniform distribution of GWs over the sky.
    """
    prob = np.log(area * (1 * u.arcsec).to_value(u.radian) ** 2 / (4 * np.pi))
    return prob

def lik_sourcepop(sgal, usenorm=False):
    """
    Computes the source population log-likelihood for a source galaxy sample.

    Parameters
    ----------
    sgal : ndarray, shape (N, 7)
        Source galaxy sample. Columns correspond to:
            0 : m_VIS_Euclid
            1 : log10_mStar
            2 : Re_maj
            3 : z
            4 : q
            5 : n_sersic
            6 : log_p_gal (from MAF)

    use_norm : bool, optional
        If True, include the normalization constant A_M* in log space:
            log p_host = log p_gal + log M_* + log(Am_star)

    Returns
    -------
    logp : ndarray, shape (N,)
        Log-likelihood of the source galaxy to emit a GW
    """
    Am_star = 1.3e-16  #eqn 21 from Wempe+ 2024
    log_p_gal = sgal[:, 6]
    M_star = 10 ** sgal[:, 1]
    if usenorm:
        logp = log_p_gal + np.log(M_star) + np.log(Am_star)
    else:
        logp = log_p_gal + np.log(M_star) 

    return logp

def lik_img(
    mag_lens, lens_reff, pos_lens,
    mag_source, reff_source, ell_source, phi_ellsource, pos_source,
    kwargs_lens,
    lens_model_class=None,
    lens_nsersic=4., source_nsersic=1.,
    elliptic_lensgal=True, lens_light_theta_ell_ell=None,
    require_source_snr=True, verbose=True
):
    """
    Calculate likelihood of a lens system being observed based on signal-to-noise.
    
    Returns
    -------
    observable : bool
        Whether the system is observable
    log_likelihood : float
        Log-likelihood (or negative penalty if unobservable)
    """
    # Compute signal-to-noise
    if lens_model_class is None:
        lens_model_class = LensModel(lens_model_list=['EPL_NUMBA', 'SHEAR'])

    SNlens, SNsource, (mag_tot_norm, mag_shear_norm) = calcStoN(
        mag_lens, lens_reff, pos_lens,
        lens_model_class, mag_source, kwargs_lens,
        reff_source, ell_source, phi_ellsource, pos_source,
        plot=False, elliptic_lensgal=elliptic_lensgal,
        require_score=True, lens_light_theta_ell_ell=lens_light_theta_ell_ell,
        lens_nsersic=lens_nsersic, source_nsersic=source_nsersic,
        verbose=verbose
    )

    SNlim = 5.0  
    if SNlens <= 0. or SNsource <= 0.:
        if verbose:
            print(f"Unobservable lens system, mlens={mag_lens}, msource={mag_source}")
        return False, -1500. + SNsource

    if SNsource < SNlim and require_source_snr:
        if verbose:
            print("Source SNR requirement not satisfied")
        return False, -1500. + SNsource
    
    return True, 0.0



def simulate_lensed_gw_detection(
    gw_params,
    lens_params,
    num_required_images=2,
    num_detected_gws=2,
    snr_threshold=7.0,
    cosmology=None,
    waveform_approximant="IMRPhenomXPHM"):

    """
    Simulate detection of a strongly lensed gravitational-wave (GW) event.

    This function does following steps:
    1) Solves the lens equation for the given lens model
    2) Computes effective lensed GW parameters for each image
    3) Calculates detector SNRs for each image
    4) Applies detection criteria based on network SNR threshold

    Parameters
    ----------
    gw_params : dict
        Dictionary containing intrinsic GW source parameters. Expected keys include:
            - mass_1, mass_2 : Detector frame component masses (solar masses)
            - theta_jn       : Inclination angle
            - psi            : Polarization angle
            - phase          : Orbital phase
            - ra, dec        : Sky location (radians)
            - geocent_time   : Geocentric merger time
            - a_1, a_2       : Dimensionless spin magnitudes
            - tilt_1, tilt_2 : Spin tilt angles
            - phi_12         : Spin azimuthal angle difference
            - phi_jl         : Azimuthal angle between total angular momentum and line of sight
        All parameters are expected in array form consistent with GWSNR input.

    lens_params : dict
        Dictionary containing lens model parameters required by the
        ImageProperties class. Typically includes:
            - zl, zs          : Lens and source redshift
            - theta_E         : Einstein radius
            - q               : Axis ratio
            - phi             : Position angle
            - gamma           : Power-law slope
            - gamma1, gamma2  : External shear components

    num_required_images : int, optional
        Minimum number of images required from the lens equation solver
        to proceed with GW parameter computation. Default is 2.

    num_detected_gws : int, optional
        Minimum number of images that must pass the SNR threshold
        for the event to be considered detected. Default is 2.

    snr_threshold : float, optional
        Network optimal SNR threshold for declaring detection
        of an individual image. Default is 7.0.

    cosmology : astropy.cosmology object, optional
        Cosmology object used for distance calculations inside
        the lensing framework. If None, a default cosmology
        inside ImageProperties is used.

    waveform_approximant : str, optional
        Waveform model used by GWSNR to compute optimal SNR.
        Default is "IMRPhenomXPHM".

    Returns
    -------
    tuple
        If lensing fails or insufficient images are produced:
            (False, None, None)

        If lensing succeeds but detection criteria fail:
            (False, snr_dict, lensed_gw_params)

        If detection criteria pass:
            (True, snr_dict, lensed_gw_params)

    snr_dict : dict
        Dictionary containing arrays of optimal SNR values
        for each retained image:
            - 'optimal_snr_net'
            - 'optimal_snr_H1'
            - 'optimal_snr_L1'
            - 'optimal_snr_V1'

    lensed_gw_params : dict
        Dictionary of effective GW parameters after lensing.
        If more images pass the SNR threshold than required,
        only the highest-SNR images are retained.
    """
    
    # Solve lens equation
    ip = ImageProperties(
        n_min_images=num_required_images,
        n_max_images=4,
        lens_model_list=['EPL_NUMBA', 'SHEAR'],
        cosmology=cosmology
    )

    try:
        lensed_output = ip.image_properties(lens_params)
    except Exception as e:
        print("Lens equation solver failed:", e)
        return False, None, None

    if lensed_output['n_images'][0] < num_required_images:
        return False, None, None

    # Produce effective GW parameters
    try:
        lensed_gw_params = ip.produce_effective_params(
            {**gw_params, **lensed_output}
        )
    except Exception as e:
        print("Effective parameter computation failed:", e)
        return False, None

    n_images = int(lensed_gw_params['n_images'][0])

    # Compute SNRs

    snr_calc = GWSNR(waveform_approximant=waveform_approximant)

    snr_net = []
    snr_H1 = []
    snr_L1 = []
    snr_V1 = []

    for i in range(n_images):

        snr_out = snr_calc.optimal_snr(
            mass_1=lensed_gw_params['mass_1'][0],
            mass_2=lensed_gw_params['mass_2'][0],
            luminosity_distance=lensed_gw_params['effective_luminosity_distance'][0, i],
            theta_jn=lensed_gw_params['theta_jn'][0],
            psi=lensed_gw_params['psi'][0],
            phase=lensed_gw_params['effective_phase'][0, i],
            geocent_time=lensed_gw_params['effective_geocent_time'][0, i],
            ra=lensed_gw_params['effective_ra'][0, i],
            dec=lensed_gw_params['effective_dec'][0, i],
            a_1=lensed_gw_params['a_1'][0],
            a_2=lensed_gw_params['a_2'][0],
            tilt_1=lensed_gw_params['tilt_1'][0],
            tilt_2=lensed_gw_params['tilt_2'][0],
            phi_12=lensed_gw_params['phi_12'][0],
            phi_jl=lensed_gw_params['phi_jl'][0]
        )

        snr_net.append(snr_out['optimal_snr_net'][0])
        snr_H1.append(snr_out['optimal_snr_H1'][0])
        snr_L1.append(snr_out['optimal_snr_L1'][0])
        snr_V1.append(snr_out['optimal_snr_V1'][0])

    snr_net = np.array(snr_net)
    snr_H1  = np.array(snr_H1)
    snr_L1  = np.array(snr_L1)
    snr_V1  = np.array(snr_V1)

    snr_dict = {
        'optimal_snr_net': snr_net,
        'optimal_snr_H1': snr_H1,
        'optimal_snr_L1': snr_L1,
        'optimal_snr_V1': snr_V1
    }

    # Detection logic
    detected_mask = snr_net >= snr_threshold
    n_detected = detected_mask.sum()

    if n_detected < num_detected_gws:
        return False, snr_dict, lensed_gw_params

    # If more pass than required,keep highest SNR ones
    if n_detected > num_detected_gws:
        sorted_idx = np.argsort(snr_net)[::-1]
        keep_idx = sorted_idx[:num_detected_gws]
    else:
        keep_idx = np.where(detected_mask)[0]


    image_keys = [
        'x0_image_positions',
        'x1_image_positions',
        'magnifications',
        'time_delays',
        'image_type',
        'effective_luminosity_distance',
        'effective_geocent_time',
        'effective_phase',
        'effective_ra',
        'effective_dec'
    ]

    for key in image_keys:
        arr = lensed_gw_params[key]
        lensed_gw_params[key] = arr[:, keep_idx]

    lensed_gw_params['n_images'][0] = len(keep_idx)

    snr_dict = {k: v[keep_idx] for k, v in snr_dict.items()}
    return True, snr_dict, lensed_gw_params

