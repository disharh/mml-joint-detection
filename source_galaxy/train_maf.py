import numpy as np
import pandas as pd
from denmarf import DensityEstimate
import matplotlib
from matplotlib import pyplot as plt
import corner

df = pd.read_csv("Final_JADES_Catalog_r1.csv")
df = df.drop(columns=['ID'])
df_clean = df.dropna()
df_clean = df_clean.reset_index(drop=True)
data_array = df_clean.to_numpy()


de = DensityEstimate().fit(
    data_array,
    num_blocks=32,
    num_hidden=128,
    num_epochs=1000,
    bounded=True,
    lower_bounds=np.array([16,4,0.001,0,0.1,0.1]),
    upper_bounds=np.array([141,12,2.5,15,1.0,8])
)

model_name = "trained_de_b32_h128_e1000_bdef_wbounds"

de.save(f"{model_name}.pkl")


xgen = de.sample(10000)
labels = ["m_VIS_Euclid", "log10(mStar)", "Re_maj", "z", "q", "n_sersic"]
lower_bounds = np.array([15, 3, 0, -1, 0, 0])
upper_bounds = np.array([141, 13, 2, 15, 1.1, 10])

ranges = [(lo, hi) for lo, hi in zip(lower_bounds, upper_bounds)]

figure = corner.corner(
    xgen,
    labels=labels,
    show_titles=True,       
    title_fmt=".2f", 
    range=ranges,
    smooth=True,        
    plot_datapoints=False,
    hist_kwargs={"density": True, "alpha": 0.5,  "color": "royalblue"},
    contour_kwargs={"colors": ["royalblue"]},
)

corner.corner(
    data_array,
    labels=labels,
    smooth=True,
    show_titles=True, 
    range=ranges,
    fig=figure, 
    plot_datapoints=False,
    hist_kwargs={"density": True, "alpha": 0.5, "color": "crimson"},
    contour_kwargs={"colors": ["crimson"]},
)

plt.savefig(f"{model_name}_corner_plot.png")

sample = pd.DataFrame(xgen, columns=labels)
sample.to_csv(f"sample_from_{model_name}.csv")
