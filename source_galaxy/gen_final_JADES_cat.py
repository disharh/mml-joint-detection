import pandas as pd

csv1 = pd.read_csv("updated_JADES_catalogs/r1/JADES_SF_mock_r1_with_VIS_Euclid.csv")
csv2 = pd.read_csv("updated_JADES_catalogs/r1/JADES_Q_mock_r1_with_VIS_Euclid.csv")

cols_to_keep = ['ID', 'm_VIS_Euclid', 'mStar', 'Re_maj', 'redshift', 'axis_ratio', 'sersic_n']

csv1_subset = csv1[cols_to_keep]
csv2_subset = csv2[cols_to_keep]

merged = pd.concat([csv1_subset, csv2_subset], ignore_index=True)

#filtered = merged[merged['m_VIS_Euclid'] < 32]

merged.to_csv("Final_JADES_Catalog_r1.csv", index=False)
