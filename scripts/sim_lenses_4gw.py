import numpy as np
import pandas as pd
import json
import time
import sys
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

job_id = int(sys.argv[1])  
np.random.seed(5678 + job_id)

num_gw = 2
n_systems = 100
n_det = 0

max_system_attempts = 100000
max_lens_attempts = 1000
max_gw_attempts = 2000

system_attempts = 0
tstart = time.perf_counter()

while n_det < n_systems and system_attempts < max_system_attempts:
    system_attempts += 1
    system_start = time.perf_counter()
    timings = {}

    ## Sample source galaxy params
    t0 = time.perf_counter()
    source_prms = sample_source_galaxy_pars()
    timings["source_sampling_time"] = time.perf_counter() - t0

    ## -------------------------
    ## LENS SAMPLING
    ## -------------------------
    t0 = time.perf_counter()
    lens_attempts = 0
    lens_success = False

    while lens_attempts < max_lens_attempts:
        lens_attempts += 1
        lens_prms = sample_lens_params(size=1, sigmazfn='cond_on_zs', zs=source_prms['z_source'])

        theta_ein = einstein_radius(
            lens_prms['sigma_lens'],
            lens_prms['z_lens'],
            source_prms['z_source']
        )

        if theta_ein >= 0.33:
            lens_prms['theta_ein'] = theta_ein
            lens_success = True
            break

    timings["lens_sampling_time"] = time.perf_counter() - t0
    timings["lens_attempts"] = lens_attempts

    if not lens_success:
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

    try:
        x_gw, y_gw, area, source_prms['x_source'], source_prms['y_source'] = sample_gwpos_then_sourcepos(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            num_detected_gws=num_gw
        )
    except Exception as e:
        print(f"Error while sampling BBH/source positions: {e}")
        continue

    ## -------------------------
    ## EM DETECTION 
    ## -------------------------
    em_start = time.perf_counter()
    em_attempts = 1  # single evaluation per system (kept explicit)

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

    em_detected = Pdet_EM[0]

    timings["em_detection_time"] = time.perf_counter() - em_start
    timings["em_attempts"] = em_attempts

    if not em_detected:
        continue

    ## -------------------------
    ## GW SAMPLING + DETECTION 
    ## -------------------------
    gw_start = time.perf_counter()
    gw_attempts = 0
    gw_success = False
    Pdet_GW = (False, None, None)

    while gw_attempts < max_gw_attempts:
        gw_attempts += 1

        gw_prms = sample_gw_params()
        z_gw = source_prms['z_source']

        gw_prms['mass_1'] = gw_prms['mass_1_source'] * (1 + z_gw)
        gw_prms['mass_2'] = gw_prms['mass_2_source'] * (1 + z_gw)

        gw_prms['luminosity_distance'] = cosmo.luminosity_distance([z_gw]).value
        gw_prms['x_gw'], gw_prms['y_gw'] = x_gw, y_gw

        Pdet_GW = simulate_lensed_gw_detection(
            gw_prms,
            kwargs_lens,
            z_lens=lens_prms['z_lens'],
            z_source=source_prms['z_source'],
            num_detected_gws=num_gw
        )

        if Pdet_GW[0]:
            gw_success = True
            break

    timings["gw_sampling_detection_time"] = time.perf_counter() - gw_start
    timings["gw_attempts"] = gw_attempts

    if not gw_success:
        print(f"[DEBUG] EM detected but GW detection failed at system_attempt={system_attempts}")
        continue

    ## -------------------------
    ## TOTAL SYSTEM TIME 
    ## -------------------------
    timings["total_system_time"] = time.perf_counter() - system_start
    timings["system_attempt"] = system_attempts

    # Save system
    n_det += 1
    print(f'System {n_det} detected!')

    gw_prms['caustic_area'] = area
    gw_prms['p_cross_sec'] = lik_cross_sec(area)
    
    source_prms_yaml = convert_dict(source_prms)
    lens_prms_yaml = convert_dict(lens_prms)
    gw_prms_yaml = convert_dict(dict(**gw_prms, **Pdet_GW[1]))

    data = {
        "source_prms": source_prms_yaml,
        "lens_prms": lens_prms_yaml,
        "gw_prms": gw_prms_yaml,
        "timings": timings
    }

    out_dir = f'lens_catalog/two_im/job_{job_id}'
    os.makedirs(out_dir, exist_ok=True)

    filename = f'{out_dir}/System_{n_det}_job{job_id}.yaml'
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

if n_det < n_systems:
    print(f"\nStopped: reached max_system_attempts = {max_system_attempts}")
    print(f"Detected {n_det}/{n_systems} systems")
else:
    print("\nAll systems successfully generated!")

t_code = time.perf_counter() - tstart
print(f'Total code execution time: {t_code}')