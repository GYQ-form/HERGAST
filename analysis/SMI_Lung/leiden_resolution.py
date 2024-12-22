import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
import HERGAST
plt.rcParams["figure.figsize"] = (15, 10)


device_idx = int(sys.argv[1])
res_list = [0.01,0.02,0.05,0.1,0.2,0.3,0.5,1,1.5,2]

# we have save this data as a h5ad file for convenience, please download at https://drive.google.com/file/d/1oe_6i3kYlawwNeZktckHd2tF1rVMK5f4/view?usp=drive_link
adata = sc.read_h5ad('SMI_lung.h5ad')

###preprocess
sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
sc.pp.scale(adata)
sc.pp.pca(adata, n_comps=200)

split = round(np.sqrt(adata.shape[0]/10000))

# HERGAST pipeline
HERGAST.utils.Cal_Spatial_Net(adata,verbose=False,k_cutoff=4)
HERGAST.utils.Cal_Expression_Net(adata, dim_reduce='PCA',verbose=False,k_cutoff=3)
train_HERGAST = HERGAST.Train_HERGAST(adata, batch_data=True, num_batch_x_y=(split,split), spatial_net_arg={'verbose':False,'k_cutoff':4},
                                      exp_net_arg={'verbose':False,'k_cutoff':3},dim_reduction='PCA',device_idx=device_idx)
train_HERGAST.train_HERGAST(save_path=None, n_epochs=200,save_loss=False,save_reconstrction=False)

# post process
sc.pp.neighbors(adata, use_rep='HERGAST')
for used_res in res_list:
    sc.tl.leiden(adata, random_state=2024, resolution=used_res,key_added='HERGAST')
    n_cluster = len(adata.obs['HERGAST'].unique())
    ARI = adjusted_rand_score(adata.obs['cell_type'], adata.obs['HERGAST'])
    NMI = normalized_mutual_info_score(adata.obs['cell_type'], adata.obs['HERGAST'])
    FMI = fowlkes_mallows_score(adata.obs['cell_type'], adata.obs['HERGAST'])
    HC = homogeneity_score(adata.obs['cell_type'], adata.obs['HERGAST'])
    print(f'resolution = {used_res} | n_cluster = {n_cluster} | ARI = {round(ARI,3)} |  NMI = {round(NMI,3)} | FMI = {round(FMI,3)} | HC = {round(HC,3)} ')
