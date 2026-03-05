### Code to sample GW and source galaxy positions using method 2 in Wempe+ 2024
### Most of it is an adapation of Ewoud's original code

import numpy as np
from lenstronomy.LensModel.Solver.epl_shear_solver import caustics_epl_shear
from sample_in_polygon import *
import SersicTransform

def get_caustics(kwargs_lens, number_images):
        """
        Return the source-plane caustic boundary for a given GW image multiplicity
        (2, 3, or 4) in an EPL+shear lens model.

        Parameters
        ----------
        kwargs_lens : list of dict
            Lens parameters in lenstronomy format for ['EPL', 'SHEAR'].
        number_images : int
            Desired image multiplicity:
            4 - quad (diamond) caustic,
            3 - inner diamond caustic,
            2 - double-image boundary (with finite magnification cutoff).

        Returns
        -------
        ndarray, shape (2, N) [default N=500 is the number of angular sampling points used to trace the caustic curve]
            Caustic coordinates (x, y) in the source plane.
        """
        maginf_cut = -1/0.1 # This shouldn't really be too bad.

        if number_images == 4:
            caustics = caustics_epl_shear(kwargs_lens, return_which='quad')
        elif number_images == 3:
            # Smallest magnification in the 3-image region outside of the diamond within the cut is very small for the central image <0.01, so these are neglected.
            caustics = caustics_epl_shear(kwargs_lens, return_which='caustic')
        elif number_images == 2:
            caustics = caustics_epl_shear(kwargs_lens, return_which='double',
                                           maginf=maginf_cut)  # Don't sample below mu=0.1. Sorta arbtirary limit. There is really no better way to do it I think.
        else:
            raise ValueError("Unsupported number of detected GWs")
        return caustics


def sample_gwpos_then_sourcepos(kwargs_lens, kwargs_source, num_detected_gws):
    """
    Sample a GW position within the caustic polygon of a lens,
    and then sample a source galaxy position around this GW using a Sersic distribution.

    Parameters
    ----------
    kwargs_lens : list of dict
        Lens parameters in lenstronomy format for model ['EPL', 'SHEAR'] 
    kwargs_source : dict
        Source galaxy parameters containing:
            - source_re : float
                Effective radius of the source galaxy.
            - source_nsersic : float
                Sersic index of the source galaxy.
            - e1_src : float
                First component of source ellipticity.
            - e2_src : float
                Second component of source ellipticity.
    num_detected_gws : int
        Number of GW images.

    Returns
    -------
    x_gw : np.float64
        Physical coordinate of the sampled GW x-position.
    y_gw : np.float64
        Physical coordinate of the sampled GW y-position.
    area : np.float64
        Area of the sampled polygon.
    source_x : np.float64
        Physical coordinate of the sampled source galaxy x-position.
    source_y : np.float64
        Physical coordinate of the sampled source galaxy y-position.

    
    Example
    -------
    kwargs_lens = [
    {
        "theta_E": 1.2,
        "gamma": 2.0,
        "e1": 0.1,
        "e2": -0.05,
        "center_x": 0.0,
        "center_y": 0.0
    },
    {
        "gamma1": 0.02,
        "gamma2": -0.01
    }
    ]

    kwargs_source = {
        "source_re": 0.5,        
        "source_nsersic": 2.5,   
        "e1_src": 0.1,
        "e2_src": 0.05           
    }

    x_gw, y_gw, area, source_x, source_y = sample_gwpos_then_sourcepos(
        kwargs_lens=kwargs_lens, 
        kwargs_source=kwargs_source, 
        num_detected_gws=2
    )
        """
    
    # Sample random unit-cube numbers for GW and source positions
    u_x_gw, u_y_gw = np.random.rand(2)
    u_x_gal, u_y_gal = np.random.rand(2)

    # Get caustic polygon for GW sampling
    poly_to_sample = get_caustics(kwargs_lens, num_detected_gws)

    # Sample GW position within the polygon
    (x_gw, y_gw), area = sample_polygon_single(poly_to_sample, np.array([u_x_gw, u_y_gw]))

    # Extract source parameters
    source_re = kwargs_source['source_re']
    source_nsersic = kwargs_source['source_nsersic']
    e1_src = kwargs_source['e1_src']
    e2_src = kwargs_source['e2_src']

    # Sample source galaxy position in Sersic coordinates around GW
    source_x, source_y = SersicTransform.sample_sersic_cart(
        [u_x_gal, u_y_gal],
        source_re,
        source_nsersic,
        e1_src,
        e2_src,
        x_gw,
        y_gw
    )
    return np.float64(x_gw), np.float64(y_gw), np.float64(area), np.float64(source_x), np.float64(source_y)