## Script for defining detection and lensing probabilities
# some functions are adapted from Ewoud's code

from image_snr_euclid import *
import numpy as np
from astropy import units as u
from ler.image_properties import ImageProperties
from gwsnr import GWSNR
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from astropy.cosmology import Planck18 as cosmo
from gw_pop import *

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
    kwargs_lens,
    z_source,
    z_lens,
    num_detected_gws,
    snr_threshold=7.0,
    waveform_approximant="IMRPhenomXPHM"):

    """
    Simulate detection of a strongly lensed gravitational-wave (GW) event.

    This function does following steps:
    1) Solves the lens equation using lenstronomy assuming EPL_NUMBA + SHEAR lens model and Planck18 cosmology
    2) Computes effective lensed GW parameters for each image
    3) Calculates detector SNRs for each image using GWSNR package
    4) Applies detection criteria based on network SNR threshold

    Parameters
    ----------
    gw_params : dict
        Dictionary containing intrinsic GW source parameters. Expected keys include:
            - x_gw, y_gw     : Angular source plane GW positions in arsecs
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

    kwargs_lens : list of dict
        Lens parameters in lenstronomy format for model ['EPL', 'SHEAR']

    z_source : float
        Source galaxy redshift

    z_lens : float
        Lens galaxy redshift

    num_detected_gws : int
        Minimum number of images that must pass the SNR threshold
        for the event to be considered detected. 

    snr_threshold : float, optional
        Network optimal SNR threshold for declaring detection
        of an individual image. Default is 7.0.

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

    """
    x_gw, y_gw = gw_params['x_gw'], gw_params['y_gw']

    # Solve lens equation
    lensModel = LensModel(lens_model_list=['EPL_NUMBA', 'SHEAR'], cosmo=cosmo)
    lensEqs = LensEquationSolver(lensModel)
    x_image, y_image = lensEqs.image_position_from_source(x_gw, y_gw, kwargs_lens, solver='analytical')
    n_images= len(x_image)

    if n_images> 5:
        print("Lens equation solver failed: 5-image solution!")
        return False, None, None

    if n_images== 0:
        print("Lens equation solver failed: 0-image solution!")
        return False, None, None
    
    if n_images<num_detected_gws:
        print("Number of images less than required!")
        return False, None, None
    
    magnifications = lensModel.magnification(x=x_image, y=y_image, kwargs=kwargs_lens)
    lensModel_withcosmo = LensModel(['EPL_NUMBA', 'SHEAR'], cosmo=cosmo, z_lens=z_lens, z_source=z_source)
    delays = lensModel_withcosmo.arrival_time(x_image=x_image, y_image=y_image, kwargs_lens=kwargs_lens) #units=days
    delays_sec = (delays - delays.min()) * 86400.0
    hessian = lensModel.hessian(x_image, y_image, kwargs_lens)
    lensed_gw_params = compute_effective_gw_params(gw_params, magnifications, delays, hessian, x_image, y_image,x_gw, y_gw)
    lensed_gw_params['magnifications'] = magnifications
    lensed_gw_params['time_delays'] = delays_sec
    lensed_gw_params['x_image'] = x_image
    lensed_gw_params['y_image'] = y_image
    lensed_gw_params['n_images'] = n_images

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
            luminosity_distance=lensed_gw_params['effective_luminosity_distance'][i],
            theta_jn=lensed_gw_params['theta_jn'][0],
            psi=lensed_gw_params['psi'][0],
            phase=lensed_gw_params['effective_phase'][i],
            geocent_time=lensed_gw_params['effective_geocent_time'][i],
            ra=lensed_gw_params['effective_ra'][i],
            dec=lensed_gw_params['effective_dec'][i],
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

    # Detection logic
    detected_mask = snr_net >= snr_threshold
    detected_indices = np.where(detected_mask)[0]
    n_detected = len(detected_indices)

    snr_dict = {
        'optimal_snr_net': snr_net,
        'optimal_snr_H1': snr_H1,
        'optimal_snr_L1': snr_L1,
        'optimal_snr_V1': snr_V1
    }

    if n_detected < num_detected_gws:
        return False, snr_dict, lensed_gw_params

    # Select exactly num_detected_gws images according to SNR 
    if n_detected > num_detected_gws:
        sorted_detected = detected_indices[np.argsort(snr_net[detected_indices])[::-1]]
        selected_indices = sorted_detected[:num_detected_gws]
    else:
        selected_indices = detected_indices

    # Filter SNRs
    snr_dict = {
        'optimal_snr_net': snr_net[selected_indices],
        'optimal_snr_H1': snr_H1[selected_indices],
        'optimal_snr_L1': snr_L1[selected_indices],
        'optimal_snr_V1': snr_V1[selected_indices]
    }

    # Filter lensed GW params
    for key, val in lensed_gw_params.items():
        if isinstance(val, np.ndarray):
            if val.ndim == 2 and val.shape[1] == n_images:
                lensed_gw_params[key] = val[:, selected_indices]
            elif val.ndim == 1 and val.shape[0] == n_images:
                lensed_gw_params[key] = val[selected_indices]

    # Update number of images
    lensed_gw_params['n_images'] = len(selected_indices)

    return True, snr_dict, lensed_gw_params      