from denmarf import DensityEstimate
import numpy as np
from lenstronomy.Util import param_util

def sample_source_galaxy_pars(size=1, model='mstar_weighted'):
    """
    Sample source galaxy parameters using a trained Masked Autoregressive Flow model
    and return the samples along with their log probability.

    Parameters
    ----------
    size : int
        Number of samples to draw.

    model : str
        Which trained MAF Model to use. Options:
        1) mstar_weighted [default] : Model trained to learn stellar mass weighted source population joint distribution
        2) non_weighted : Model trained to learn intrinsic source population joint distribution

    Returns
    -------
    sgal : dict
        Dictionary of arrays (float if size=1) containing sampled source galaxy parameters:
            'm_VIS_Euclid' - Magnitude in Euclid VIS band
            'log10_mStar' - Stellar mass in log10 solar mass
            'Re_maj_source' - Sersic effective radius in arcsec
            'z_source' - source galaxy redshift
            'q_source' - axis ratio
            'n_sersic_source' - Sersic index
            'log_p_source' - Log probability of the sampled galaxy under the trained MAF
            'theta_light_source' - axis angle (in radians)
            'e1_source' - complex ellipticity component 1
            'e2_source' - complex ellipticity component 2
    """

    if model=='mstar_weighted':
        trained_de = "source_galaxy/trained_de_weighted_mass_b32_h128_e1000_sizen.pkl"
    elif model=='non_weighted':
        trained_de = "source_galaxy/trained_de_b32_h128_e1000_bdef_wbounds.pkl"
    else:
        raise ValueError('Invalid model name!')

    de = DensityEstimate.from_file(filename=trained_de)
    xgen = de.sample(size)  
    log_p = de.score_samples(xgen) 
    
    # ellipticity angle is isotropically distributed
    theta_light_source = np.random.uniform(0.0, np.pi,size)

    e1_source, e2_source = param_util.phi_q2_ellipticity(
        theta_light_source,
        xgen[:, 4]
    )

    sgal = {
        'm_VIS_Euclid': xgen[:, 0],
        'log10_mStar': xgen[:, 1],
        'Re_maj_source': xgen[:, 2],
        'z_source': xgen[:, 3],
        'q_source': xgen[:, 4],
        'n_sersic_source': xgen[:, 5],
        'log_p_source': log_p,
        'theta_light_source' : theta_light_source, 
        'e1_source': e1_source,
        'e2_source' : e2_source
    }

    
    if size == 1:
        sgal = {k: float(v[0]) for k, v in sgal.items()}

    return sgal