import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.manifold import TSNE
import umap

print('Loading data...')
#input_path = 'data/PANCAN/GDC-PANCAN_'
input_path='/home/ldap/ltoure/OmiVAE/results/OmiVAE_expr128D_128D_'

sample_id = np.loadtxt(input_path + 'both_samples.tsv', delimiter='\t', dtype='str')

input_df = pd.read_csv(input_path + 'latent_space.tsv', sep='\t', header=0, index_col=0)
input_df = input_df.T

latent_space_dimension = 2

# PCA
print('PCA')
pca = decomposition.PCA(n_components=latent_space_dimension)
z = pca.fit_transform(input_df.values)
latent_code = pd.DataFrame(z, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_PCA_latent_sapce.tsv'
latent_code.to_csv('/home/ldap/ltoure/OmiVAE/GDC-PANCAN_2D_PCA_latent_sapce.tsv', sep='\t')


input_df['y'] = traitData['response']
import matplotlib.pyplot as plt
import seaborn as sns
input_df['pca-one'] = z[:,0]
input_df['pca-two'] = z[:,1] 

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("dark", 2),
    data=input_df,
    legend="full",
    alpha=0.8
)
plt.savefig('/home/ldap/ltoure/OmiVAE/results/OmiVAe_pca.png')

# KPCA
print('KPCA')
kpca = decomposition.KernelPCA(n_components=latent_space_dimension, kernel='rbf')
z = kpca.fit_transform(input_df.values)
latent_code = pd.DataFrame(z, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_KPCA_latent_sapce.tsv'
latent_code.to_csv('/home/ldap/ltoure/OmiVAE/GDC-PANCAN_2D_KPCA_latent_sapce.tsv', sep='\t')

input_df['kpca-one'] = z[:,0]
input_df['kpca-two'] = z[:,1] 

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="kpca-one", y="kpca-two",
    hue="y",
    palette=sns.color_palette("dark", 2),
    data=input_df,
    legend="full",
    alpha=0.8
)
plt.savefig('/home/ldap/ltoure/OmiVAE/results/OmiVAe_kpca.png')


# TSNE
print('TSNE')
tsne = TSNE(n_components=latent_space_dimension)
z = tsne.fit_transform(input_df.values)
latent_code = pd.DataFrame(z, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_TSNE_latent_sapce.tsv'
latent_code.to_csv(output_path, sep='\t')

# TSNE
print('TSNE')
tsne = TSNE(n_components=latent_space_dimension)
tsne_results = tsne.fit_transform(input_df.values)
latent_code = pd.DataFrame(z, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_TSNE_latent_sapce.tsv'
latent_code.to_csv('/home/ldap/ltoure/OmiVAE/GDC-PANCAN_2D_TSNE_latent_sapce.tsv', sep='\t')

import maplotlib.pyplot as plt
import seaborn as sns

input_df.index=traitData.index
input_df['y'] = traitData['response']

input_df['tsne-2d-one'] = tsne_results[:,0]
input_df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("dark", 2),
    data=input_df,
    legend="full",
    alpha=0.8
)
plt.savefig('/home/ldap/ltoure/OmiVAE/results/OmiVAe_tsne.png')

#pip install umap-learn
import umap.umap_ as umap
# UMAP
print('UMAP')
umap_reducer = umap.UMAP()
umap_results = umap_reducer.fit_transform(input_df.values)
latent_code = pd.DataFrame(umap_results, index=sample_id)
output_path = 'results/GDC-PANCAN_' + str(latent_space_dimension) + 'D_UMAP_latent_sapce.tsv'
latent_code.to_csv(output_path, sep='\t')



input_df['umap-2d-one'] = umap_results[:,0]
input_df['umap-2d-two'] = umap_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="umap-2d-one", y="umap-2d-two",
    hue="y",
    palette=sns.color_palette("dark", 2),
    data=input_df,
    legend="full",
    alpha=0.8
)
plt.savefig('/home/ldap/ltoure/OmiVAE/results/OmiVAe_umap.png')
