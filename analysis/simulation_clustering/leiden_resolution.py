import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import sys
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
import math
import HERGAST
plt.rcParams["figure.figsize"] = (10, 10)

device_idx = sys.argv[1]
res_list = [0.01,0.02,0.05,0.1,0.2,0.3,0.5,1]

print(f'##############################      HERGAST       ############################')

adata = sc.read_h5ad('data.h5ad') # replace to the simulated data directory
split = max(round(np.sqrt(adata.shape[0]/10000)),1)

# preprocess
sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
sc.pp.scale(adata)
sc.pp.pca(adata, n_comps=200)

####### HERGAST
HERGAST.utils.Cal_Spatial_Net(adata,verbose=False)
HERGAST.utils.Cal_Expression_Net(adata, dim_reduce='PCA',verbose=False)
global train_HERGAST
train_HERGAST = HERGAST.Train_HERGAST(adata, batch_data=True if split > 1 else False, num_batch_x_y=(split,split),dim_reduction='PCA',spatial_net_arg={'verbose':False},
                            device_idx = device_idx, exp_net_arg={'verbose':False})
train_HERGAST.train_HERGAST(save_path=None, n_epochs=200,save_loss=False,save_reconstrction=False)

# post process
sc.pp.neighbors(adata, use_rep='HERGAST')
for used_res in res_list:
    sc.tl.leiden(adata, random_state=2024, resolution=used_res,key_added='HERGAST')
    n_cluster = len(adata.obs['HERGAST'].unique())
    ARI = adjusted_rand_score(adata.obs['ann_level_3'], adata.obs['HERGAST'])
    NMI = normalized_mutual_info_score(adata.obs['ann_level_3'], adata.obs['HERGAST'])
    FMI = fowlkes_mallows_score(adata.obs['ann_level_3'], adata.obs['HERGAST'])
    HC = homogeneity_score(adata.obs['ann_level_3'], adata.obs['HERGAST'])
    print(f'resolution = {used_res} | n_cluster = {n_cluster} | ARI = {round(ARI,3)} |  NMI = {round(NMI,3)} | FMI = {round(FMI,3)} | HC = {round(HC,3)} ')
