import numpy as np
import pandas as pd
import json
import time
import yaml

from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from lenstronomy.Util import param_util

from lens_mass import *
from lens_light import *
from source_light import *
from gw_pop import *
from bbh_pos import *
from likelihood import *
from utils import *

num_gw = 2
n_systems = 100
n = 0

results = []  

while n < n_systems:

    source_prms = sample_source_galaxy_pars()
    lens_prms = sample_lens_params()

    if source_prms['z_source'] <= lens_prms['z_lens']:
        continue

    theta_ein = einstein_radius(
        lens_prms['sigma_lens'],
        lens_prms['z_lens'],
        source_prms['z_source']
    )

    lens_prms['theta_ein'] = theta_ein

    kwargs_lens = [
        {
            'theta_E': lens_prms['theta_ein'],
            'gamma': lens_prms['gamma'],
            'e1': lens_prms['e1_lens'],
            'e2': lens_prms['e2_lens'],
            'center_x': lens_prms['x_lens'],
            'center_y': lens_prms['y_lens']
        },
        {
            'gamma1': lens_prms['gamma1'],
            'gamma2': lens_prms['gamma2']
        }
    ]

    kwargs_source = {
        "re_source": source_prms["Re_maj_source"],
        "nsersic_source": source_prms["n_sersic_source"],
        "e1_source": source_prms["e1_source"],
        "e2_source": source_prms["e2_source"],
    }

    x_gw, y_gw, area, source_prms['x_source'], source_prms['y_source'] = sample_gwpos_then_sourcepos(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        num_detected_gws=num_gw
    )

    Pdet_EM = lik_img(
        lens_prms['mag_lens'],
        lens_prms['re_lens'],
        (lens_prms['x_lens'], lens_prms['y_lens']),
        source_prms['m_VIS_Euclid'],
        kwargs_source['re_source'],
        1 - source_prms['q_source'],
        source_prms['theta_light_source'],
        (source_prms['x_source'], source_prms['y_source']),
        kwargs_lens,
        lens_model_class=None,
        lens_nsersic=4.,
        source_nsersic=kwargs_source['nsersic_source'],
        elliptic_lensgal=True,
        lens_light_theta_ell_ell=None,
        require_source_snr=True,
        verbose=True
    )

    det = 1 if Pdet_EM[0] else 0

    results.append({
        # lens params
        "sigma_lens": lens_prms["sigma_lens"],
        "z_lens": lens_prms["z_lens"],
        "theta_ein": lens_prms["theta_ein"],
        "gamma": lens_prms["gamma"],
        "e1_lens": lens_prms["e1_lens"],
        "e2_lens": lens_prms["e2_lens"],
        "x_lens": lens_prms["x_lens"],
        "y_lens": lens_prms["y_lens"],
        "mag_lens": lens_prms["mag_lens"],
        "re_lens": lens_prms["re_lens"],

        # source params
        "z_source": source_prms["z_source"],
        "x_source": source_prms["x_source"],
        "y_source": source_prms["y_source"],
        "Re_maj_source": source_prms["Re_maj_source"],
        "n_sersic_source": source_prms["n_sersic_source"],
        "e1_source": source_prms["e1_source"],
        "e2_source": source_prms["e2_source"],
        "q_source": source_prms["q_source"],
        "theta_light_source": source_prms["theta_light_source"],
        "m_VIS_Euclid": source_prms["m_VIS_Euclid"],

        # GW info
        "num_gw": num_gw,
        "area": area,

        # detection
        "det": det
    })

    n += 1

df = pd.DataFrame(results)
df.to_csv("lens_em_systems_catalog_for_training.csv", index=False)
