import yaml
import glob
import pandas as pd
import numpy as np
import corner
import matplotlib.pyplot as plt
from lenstronomy.Util import param_util
from pathlib import Path

source_keys = [
    "z_source",
    "m_VIS_Euclid",
    "log10_mStar",
    "Re_maj_source",
    "q_source",
    "n_sersic_source"
]

lens_keys = [
    "z_lens",
    "sigma_lens",
    "q_lens",
    "theta_ein",
    "gamma",
    "gamma1",
    "gamma2"
]


base_dir = Path("/home/disha.hegde/projects/mml/github/mml-joint-detection/lens_catalog/four_im")
yaml_files = list(base_dir.rglob("*.yaml"))
print(f"Found {len(yaml_files)} yaml files")

source_rows = []
lens_rows = []

for fname in yaml_files:

    with open(fname, "r") as f:
        data = yaml.safe_load(f)

    source = data["source_prms"]
    lens = data["lens_prms"]

    source_rows.append(
        {k: source[k] for k in source_keys}
    )

    lens_rows.append(
        {k: lens[k] for k in lens_keys}
    )


source_df = pd.DataFrame(source_rows)
lens_df = pd.DataFrame(lens_rows)

data_source = np.column_stack([
    source_df['z_source'],
    source_df['log10_mStar'],
    1 - source_df['q_source'],
    source_df['Re_maj_source'],
    source_df['m_VIS_Euclid']
])

lower_bounds_s = np.array([0, 5.8, 0, 0, 19])
upper_bounds_s = np.array([10, 13, 1, 2, 32])

ranges_s = [(lo, hi) for lo, hi in zip(lower_bounds_s, upper_bounds_s)]

figure = corner.corner(
    data_source,
    labels=['$z_s$', '$log_{10} \, (M_{\star}/M_{\odot})$', '$1-q_s$', '$R_s$', '$m_s$'],
    show_titles=True,
    title_fmt=".2f", 
    range=ranges_s,
    smooth=True,        
    plot_datapoints=False,
    hist_kwargs={"density": True, "alpha": 0.5,  "color": "royalblue"},
    contour_kwargs={"colors": ["royalblue"]},
)

figure.savefig(f"{base_dir}/source_distributions_corner.png", dpi=400)

phi_s, gamma_s = param_util.shear_cartesian2polar(lens_df['gamma1'],lens_df['gamma2'])
data_lens = np.column_stack([
    lens_df['theta_ein'],
    lens_df['sigma_lens'],
    1 - lens_df['q_lens'],
    gamma_s,
    lens_df['z_lens'],
    lens_df['gamma']
])

lower_bounds_l = np.array([0, 0, 0, 0, 0, 1.0])
upper_bounds_l = np.array([3, 600, 1, 0.25, 3, 2.6])

ranges_l = [(lo, hi) for lo, hi in zip(lower_bounds_l, upper_bounds_l)]

figure = corner.corner(
    data_lens,
    labels=['$\Theta_E$ (arcsec)', '$\sigma_l$ (km/s)', '$1-q_l$', '$\gamma_s$', '$z_l$', "$\gamma_m$"],   
    show_titles=True,       
    title_fmt=".2f", 
    range=ranges_l,
    smooth=True,        
    plot_datapoints=False,
    hist_kwargs={"density": True, "alpha": 0.5,  "color": "green"},
    contour_kwargs={"colors": ["green"]},
)


figure.savefig(f"{base_dir}/lens_distributions_corner.png", dpi=400)
