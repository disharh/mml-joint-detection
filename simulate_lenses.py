import numpy as np
import pandas as pd
import json
import time
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
n_systems = 2
n_det = 0

max_system_attempts = 50
max_lens_attempts = 100
max_gw_attempts = 100

system_attempts = 0

while n_det < n_systems and system_attempts < max_system_attempts:
    system_attempts += 1
    system_start = time.perf_counter()
    timings = {}

    ## Sample source galaxy params
    t0 = time.perf_counter()
    source_prms = sample_source_galaxy_pars()
    timings["source_sampling_time"] = time.perf_counter() - t0

    ## Sample lens galaxy params 
    # fast rejection if z_s <= z_l and einstein radius < 0.33
    t0 = time.perf_counter()
    lens_attempts = 0
    lens_success = False

    while lens_attempts < max_lens_attempts:
        lens_attempts += 1
        lens_prms = sample_lens_params()

        if source_prms['z_source'] <= lens_prms['z_lens']:
            continue

        theta_ein = einstein_radius(
            lens_prms['sigma_lens'],
            lens_prms['z_lens'],
            source_prms['z_source']
        )

        if theta_ein >= 0.33:
            lens_prms['theta_ein'] = theta_ein
            break

    timings["lens_sampling_time"] = time.perf_counter() - t0
    timings["lens_attempts"] = lens_attempts

    if not lens_success:
        print("Lens sampling failed: max attempts reached")
        continue

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

    ## Sample GW params and BBH+source galaxy positions till 
    # required num of GW images are detectable
    t0 = time.perf_counter()
    gw_attempts = 0
    gw_success = False
    Pdet_GW=(False, None, None)

    while gw_attempts < max_gw_attempts:
        gw_attempts += 1
        gw_prms = sample_gw_params()
        z_gw = source_prms['z_source']

        gw_prms['mass_1'] = gw_prms['mass_1_source'] * (1 + z_gw)
        gw_prms['mass_2'] = gw_prms['mass_2_source'] * (1 + z_gw)

        gw_prms['luminosity_distance'] = cosmo.luminosity_distance([z_gw]).value
        gw_prms['x_gw'], gw_prms['y_gw'], area, source_prms['x_source'], source_prms['y_source'] = sample_gwpos_then_sourcepos(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            num_detected_gws=num_gw
        )

        Pdet_GW = simulate_lensed_gw_detection(gw_prms, kwargs_lens, z_lens=lens_prms['z_lens'], z_source=source_prms['z_source'], num_detected_gws=num_gw)
        if Pdet_GW[0]:
            gw_success = True
            break

    timings["gw_sampling_detection_time"] = time.perf_counter() - t0
    timings["gw_attempts"] = gw_attempts
    if not gw_success:
        print("GW detection failed: max attempts reached")
        continue

    ## Check EM detectability
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
    
    detected = Pdet_EM[0]
    timings["total_system_time"] = time.perf_counter() - system_start
    timings["system_attempt"] = system_attempts

    # Save system to YAML if EM detectable
    if detected:
        n_det = n_det+1
        print(f'System {n_det} detected!')

        source_prms_yaml = convert_dict(source_prms)
        lens_prms_yaml = convert_dict(lens_prms)
        gw_prms_yaml = convert_dict(dict(**gw_prms, **Pdet_GW[1]))

        data = {
        "source_prms": source_prms_yaml,
        "lens_prms": lens_prms_yaml,
        "gw_prms": gw_prms_yaml,
        "timings": timings
        }

        filename = f'System_{n_det}.yaml'
        with open(filename, "w") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    else:
        print(f'Not detected!')
        continue

if n_det < n_systems:
    print(f"\nStopped: reached max_system_attempts = {max_system_attempts}")
    print(f"Detected {n_det}/{n_systems} systems")
else:
    print("\nAll systems successfully generated!")