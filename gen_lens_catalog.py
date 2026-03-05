import numpy as np
import pandas as pd
import json
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from lenstronomy.Util import param_util

from lens_mass import *
from lens_light import *
from source_light import *
from gw_pop import *
from bbh_pos import *
from likelihood import *


# number of samples
N_SAMPLES = 1000


detected_rows = []
not_detected_rows = []


for i in range(N_SAMPLES):

    print(f"Running sample {i}")

    # -----------------------------
    # SAMPLE SOURCE GALAXY
    # -----------------------------
    source_prms = sample_source_galaxy_pars()

    zs = source_prms[0][3]
    source_q = source_prms[0][4]
    source_theta_ell = np.random.uniform(0.0, np.pi)

    e1_src, e2_src = param_util.phi_q2_ellipticity(
        source_theta_ell,
        source_q
    )

    kwargs_source = {
        "source_re": float(source_prms[0][2]),
        "source_nsersic": float(source_prms[0][5]),
        "e1_src": e1_src,
        "e2_src": e2_src
    }

    # -----------------------------
    # SAMPLE LENS PARAMETERS
    # -----------------------------
    sigma, zl = sample_sigmaz_ler()

    ell_light, theta_light, ell_mass, theta_mass = sample_ellipticity_theta(
        sigma=sigma,
        size=1
    )

    gamma_slope = sample_slope_gamma()

    gamma_shear, phi_shear = sample_shear()

    lens_x, lens_y = sample_lens_position()

    Mr, re, k_corr = sample_FP(sigma, zl, ell_light)

    theta_ein = einstein_radius(float(sigma), float(zl), float(zs))

    mag_lens = Mr + 5 * np.log10(
        (cosmo.luminosity_distance(zl) / (10 * u.pc)).to_value(1)
    ) + k_corr

    e1, e2 = param_util.phi_q2_ellipticity(
        phi=theta_mass,
        q=1 - ell_mass
    )

    gamma1, gamma2 = param_util.shear_polar2cartesian(
        phi=phi_shear,
        gamma=gamma_shear
    )

    kwargs_lens = [
        {
            'theta_E': float(theta_ein),
            'gamma': float(gamma_slope),
            'e1': float(e1),
            'e2': float(e2),
            'center_x': float(lens_x),
            'center_y': float(lens_y)
        },
        {
            'gamma1': float(gamma1),
            'gamma2': float(gamma2)
        }
    ]

    lens_params = {
        'zl': zl,
        'zs': np.array([zs]),
        'theta_E': np.array([theta_ein]),
        'q': 1 - ell_mass,
        'phi': theta_mass,
        'gamma': gamma_slope,
        'gamma1': gamma1,
        'gamma2': gamma2
    }

    # -----------------------------
    # SAMPLE GW PARAMETERS
    # -----------------------------
    gw_prms = sample_gw_params(1)

    zgw = zs

    gw_prms['mass_1'] = gw_prms['mass_1_source'] * (1 + zgw)
    gw_prms['mass_2'] = gw_prms['mass_2_source'] * (1 + zgw)

    gw_prms['luminosity_distance'] = cosmo.luminosity_distance([zgw]).value

    # -----------------------------
    # SAMPLE POSITIONS
    # -----------------------------
    x_gw, y_gw, area, source_x, source_y = sample_gwpos_then_sourcepos(
        kwargs_lens=kwargs_lens,
        kwargs_source=kwargs_source,
        num_detected_gws=4
    )

    # -----------------------------
    # PROBABILITIES
    # -----------------------------
    pcross = float(lik_cross_sec(area))

    psrc = float(lik_sourcepop(source_prms))

    PdetEM = lik_img(
        float(mag_lens),
        float(re),
        (float(lens_x), float(lens_y)),
        float(source_prms[0][0]),
        kwargs_source['source_re'],
        1 - float(source_q),
        float(source_theta_ell),
        (source_x, source_y),
        kwargs_lens,
        lens_model_class=None,
        lens_nsersic=4.,
        source_nsersic=kwargs_source['source_nsersic'],
        elliptic_lensgal=True,
        lens_light_theta_ell_ell=None,
        require_source_snr=True,
        verbose=True
    )


    PdetGW = simulate_lensed_gw_detection(
        gw_prms,
        lens_params,
        num_required_images=2,
        num_detected_gws=2,
        snr_threshold=7.0
    )

    detected = PdetEM[0] and PdetGW[0]

    row = {}

    row["detected"] = detected

    # SOURCE PARAMETERS
    row.update({
        "m_VIS_Euclid": float(source_prms[0][0]),
        "log10_mStar": float(source_prms[0][1]),
        "Re_maj": float(source_prms[0][2]),
        "z_source": float(source_prms[0][3]),
        "q_source": float(source_prms[0][4]),
        "n_sersic_source": float(source_prms[0][5]),
        "log_p_gal": float(source_prms[0][6]),
        "source_theta_ell": float(source_theta_ell),
        "source_x": float(source_x),
        "source_y": float(source_y)
    })

    # GW PARAMETERS
    for key in [
        'mass_1','mass_2','theta_jn','psi','phase','ra','dec',
        'geocent_time','a_1','a_2','tilt_1','tilt_2','phi_12','phi_jl'
    ]:
        row[key] = float(gw_prms[key])

    row["luminosity_distance"] = float(gw_prms['luminosity_distance'])

    # PROBABILITIES
    row["pcross"] = pcross
    row["psrc"] = psrc

    row["PdetEM_flag"] = PdetEM[0]
    row["PdetEM_logL"] = PdetEM[1]
    row["PdetGW_flag"] = PdetGW[0]

    # -----------------------------
    # SAVE SNR
    # -----------------------------
    if PdetGW[1] is not None:

        snr_dict = PdetGW[1]

        row["snr_net"] = json.dumps(snr_dict["optimal_snr_net"].tolist())
        row["snr_H1"] = json.dumps(snr_dict["optimal_snr_H1"].tolist())
        row["snr_L1"] = json.dumps(snr_dict["optimal_snr_L1"].tolist())
        row["snr_V1"] = json.dumps(snr_dict["optimal_snr_V1"].tolist())

    else:

        row["snr_net"] = None
        row["snr_H1"] = None
        row["snr_L1"] = None
        row["snr_V1"] = None

    # -----------------------------
    # SAVE LENSED GW PARAMETERS
    # -----------------------------
    if PdetGW[2] is not None:

        lensed = PdetGW[2]

        for key in [
            "x0_image_positions",
            "x1_image_positions",
            "magnifications",
            "time_delays",
            "image_type",
            "effective_luminosity_distance",
            "effective_geocent_time",
            "effective_phase",
            "effective_ra",
            "effective_dec"
        ]:

            row[key] = json.dumps(lensed[key].tolist())

    else:

        for key in [
            "x0_image_positions",
            "x1_image_positions",
            "magnifications",
            "time_delays",
            "image_type",
            "effective_luminosity_distance",
            "effective_geocent_time",
            "effective_phase",
            "effective_ra",
            "effective_dec"
        ]:

            row[key] = None

    # -----------------------------
    # SAVE TO CORRECT LIST
    # -----------------------------
    if detected:
        detected_rows.append(row)
        print("Detected!")
    else:
        not_detected_rows.append(row)
        print("Not detected")


# -----------------------------
# SAVE CATALOGS
# -----------------------------
detected_df = pd.DataFrame(detected_rows)
not_detected_df = pd.DataFrame(not_detected_rows)

detected_df.to_csv("detected_catalog.csv", index=False)
not_detected_df.to_csv("not_detected_catalog.csv", index=False)

print("Catalogs saved.")