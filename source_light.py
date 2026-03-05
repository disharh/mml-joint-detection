from denmarf import DensityEstimate
import numpy as np

def sample_source_galaxy_pars(size=1):
    """
    Sample source galaxy parameters using a trained Masked Autoregressive Flow model
    and return the samples along with their log probability under the galaxy population.

    Parameters
    ----------
    size : int
        Number of samples to draw.

    Returns
    -------
    xgen : ndarray, shape (size, 7)
        Array of sampled galaxy parameters. Columns correspond to:
            0 : m_VIS_Euclid (Magnitude in Euclid VIS band)
            1 : log10_mStar (Stellar mass in log10 solar mass)
            2 : Re_maj (Sersic effective radius in arcsec)
            3 : z (source galaxy redshift)
            4 : q (axis ratio)
            5 : n_sersic (Sersic index)
            6 : log_p_gal (Log probability of the sampled galaxy under the trained MAF)
    """

    trained_de = "source_galaxy/trained_de_b32_h128_e1000_bdef_wbounds.pkl"
    de = DensityEstimate.from_file(filename=trained_de)

    xgen = de.sample(size)  
    log_p_gal = de.score_samples(xgen) 

    sgal = np.hstack([xgen, log_p_gal[:, None]])

    return sgal