from ler.lens_galaxy_population import LensGalaxyParameterDistribution
import numpy as np
import matplotlib.pyplot as plt


lens_param_samplers = dict(
    velocity_dispersion="velocity_dispersion_ewoud"
)

lens_param_samplers_params = {
    "velocity_dispersion": {
        "sigma_min": 60,    # km/s
        "sigma_max": 600    # km/s
    }
}

lens = LensGalaxyParameterDistribution(
    z_min=0.0,
    z_max=3.0,                     
    lens_param_samplers=lens_param_samplers,
    lens_param_samplers_params=lens_param_samplers_params,
    create_new_interpolator=False
)

params = lens.sample_all_routine_epl_shear_intrinsic(size=10000)

zl = params["zl"]
sigma = params["sigma"]


print("\nSample summary")
print("--------------------------")
print("Number of samples:", len(zl))
print("Lens redshift range:", zl.min(), zl.max())
print("Velocity dispersion range:", sigma.min(), sigma.max())
print("Mean zl:", np.mean(zl))
print("Mean sigma:", np.mean(sigma))


plt.figure()
plt.hist(zl, bins=60)
plt.xlabel("Lens redshift zl")
plt.ylabel("Count")
plt.title("Intrinsic Lens Redshift Distribution")

plt.figure()
plt.hist(sigma, bins=60)
plt.xlabel("Velocity dispersion σ [km/s]")
plt.ylabel("Count")
plt.title("Ewoud Velocity Dispersion Function")

plt.savefig('test_ler_sigmaz_1.png')
