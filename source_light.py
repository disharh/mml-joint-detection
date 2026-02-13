from denmarf import DensityEstimate

def sample_source_galaxy_pars(size):
    """
    Sample source galaxy parameters using a trained Masked Autoregressive Flow model.

    Parameters
    ----------
    size : int
        Number of samples to draw.

    Returns
    -------
    m_VIS_Euclid : Magnitude in Euclid VIS band (ndarray)
    log10_mStar : Stellar mass in log10 solar mass (ndarray)
    Re_maj : Sersic (effective) radius in kPc (ndarray)
    z : Source galaxy redshift (ndarray)
    q : Axis ratio (ndarray)
    n_sersic : Sersic index (ndarray)
    
    Each array has length `size`.
    """

    trained_de = "source_galaxy/trained_de_b32_h128_e1000_bdef_wbounds.pkl"
    de = DensityEstimate.from_file(filename=trained_de)

    xgen = de.sample(size)  # shape: (size, 6)

    m_VIS_Euclid = xgen[:, 0]
    log10_mStar = xgen[:, 1]
    Re_maj = xgen[:, 2]
    z = xgen[:, 3]
    q = xgen[:, 4]
    n_sersic = xgen[:, 5]

    return m_VIS_Euclid, log10_mStar, Re_maj, z, q, n_sersic