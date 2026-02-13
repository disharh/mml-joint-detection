import numpy as np
from astropy.io import fits
import pandas as pd
import os

##Star-forming JADES galaxies
#spec_dir = "/home/disha.hegde/projects/mml/jaguar_mock_catalog/JADES_SF_spec/r1/"
#catalog_file = "/home/disha.hegde/projects/mml/jaguar_mock_catalog/JADES_SF_mock_r1_v1.2.fits"
#spec_files = os.listdir(spec_dir)

##Quiescient JADES galaxies
spec_dir = "/home/disha.hegde/projects/mml/jaguar_mock_catalog/JADES_Q_spec/"
catalog_file = "/home/disha.hegde/projects/mml/jaguar_mock_catalog/JADES_Q_mock_r1_v1.2.fits"
spec_files = ["/home/disha.hegde/projects/mml/jaguar_mock_catalog/JADES_Q_spec/JADES_Q_mock_r1_v1.2_spec_5A_30um.fits"]

euclid_filter_file = "/home/disha.hegde/projects/mml/Euclid_VIS.vis.dat"
output_dir = "/home/disha.hegde/projects/mml/updated_JADES_catalogs/r1/"
os.makedirs(output_dir, exist_ok=True)

# Speed of light in Å/s 
c = 2.99792458e18

# Load Euclid VIS filter curve
euclid_data = np.loadtxt(euclid_filter_file)
euclid_wav = euclid_data[:, 0]       # Angstroms
euclid_trans = euclid_data[:, 1]     # Transmission


# Function to compute band-integrated flux in f_lambda
def compute_band_flux(wav, flux, filt_wav, filt_trans):
    f_interp = np.interp(wav, filt_wav, filt_trans, left=0, right=0)
    dlam = np.gradient(wav)
    num = np.sum(flux * f_interp * dlam)
    den = c * np.sum(f_interp * (1.0 / wav**2) * dlam)
    f_nu = num / den if den > 0 else np.nan
    return f_nu

# Load the main catalog
with fits.open(catalog_file, mode="readonly") as hdul:
    main_cat = hdul[1].data
    col_names = hdul[1].columns.names
    #print(main_cat)
    main_df = pd.DataFrame(main_cat, columns=col_names)
    #main_df = pd.DataFrame(np.array(main_cat).byteswap().newbyteorder(), columns=col_names)

main_df["f_nu_VIS_Euclid"] = np.nan
main_df["m_VIS_Euclid"] = np.nan

print(main_df["ID"])
# Process each spectral FITS file
for spec_file in spec_files:
    s_file = os.path.join(spec_dir, spec_file)
    print(f"Processing {os.path.basename(s_file)} ...")
    with fits.open(s_file, memmap=True) as hdul:
        f_rest = hdul[1].data.T      
        lam_rest = hdul[2].data     # Angstroms
        obj_props = hdul[3].data
        IDs = obj_props["ID"]
        z = obj_props["redshift"]

    for i in range(len(IDs)):
        gal_id = IDs[i]
        redshift = z[i]
        print(gal_id)
        # Convert to observed frame
        lam_obs = lam_rest * (1 + redshift)
        flux_obs = f_rest[:, i] / (1 + redshift)   # f_lambda, observed

        # Integrate using Euclid VIS filter
        f_nu= compute_band_flux(lam_obs, flux_obs, euclid_wav, euclid_trans)
        
        # AB magnitude
        m_vis = -2.5 * np.log10(f_nu) - 48.6 if f_nu > 0 else np.nan
        
        # Update main catalog
        main_df.loc[main_df["ID"] == gal_id, "f_nu_VIS_Euclid"] = f_nu
        main_df.loc[main_df["ID"] == gal_id, "m_VIS_Euclid"] = m_vis

        # Save partial catalog after each file
        if len(spec_files)>1:
            part_out = os.path.join(output_dir, f"updated_catalog_{os.path.basename(s_file).split('.fits')[0]}.csv")
            main_df.to_csv(part_out, index=False)
            print(f"Saved {part_out}")


#Get some other quantities
main_df["M_VIS_Euclid"] = main_df["m_VIS_Euclid"] - 5*np.log10(main_df["luminosity_distance"]*1e5)
main_df["mStar_lum"] =  main_df["mStar"] - 2*np.log10(main_df["luminosity_distance"]*1e5)
main_df["log10Re"] = np.log10(main_df["Re_maj"])
main_df["log1z"] = np.log(1 + main_df["redshift"])

# Save the final catalog
#final_out = os.path.join(output_dir, "JADES_SF_mock_r1_with_VIS_Euclid.csv")
final_out = os.path.join(output_dir, "JADES_Q_mock_r1_with_VIS_Euclid.csv")
main_df.to_csv(final_out, index=False)
print(f"\n Final catalog saved to: {final_out}")
