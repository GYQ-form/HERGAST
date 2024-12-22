import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import math
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
import HERGAST
plt.rcParams["figure.figsize"] = (15, 10)


device_idx = int(sys.argv[1])
PCA_dims = [50,100,150,200,250,300,350,400,450,500]

for pc in PCA_dims:
    # we have save this data as a h5ad file for convenience, please download at https://drive.google.com/file/d/1oe_6i3kYlawwNeZktckHd2tF1rVMK5f4/view?usp=drive_link
    adata = sc.read_h5ad('SMI_lung.h5ad')

    ###preprocess
    sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=pc)

    split = math.ceil(np.sqrt(adata.shape[0]/10000)*(pc/200))

    # HERGAST pipeline
    HERGAST.utils.Cal_Spatial_Net(adata,verbose=False,k_cutoff=4)
    HERGAST.utils.Cal_Expression_Net(adata, dim_reduce='PCA',verbose=False,k_cutoff=3)
    train_HERGAST = HERGAST.Train_HERGAST(adata, batch_data=True, num_batch_x_y=(split,split), spatial_net_arg={'verbose':False,'k_cutoff':4},
                                        exp_net_arg={'verbose':False,'k_cutoff':3},dim_reduction='PCA',device_idx=device_idx)
    train_HERGAST.train_HERGAST(save_path=None, n_epochs=200,save_loss=False,save_reconstrction=False)

    # clustering & evaluation
    sc.pp.neighbors(adata, use_rep='HERGAST')
    sc.tl.leiden(adata, random_state=2024, resolution=0.3,key_added='HERGAST')
    ARI = adjusted_rand_score(adata.obs['cell_type'], adata.obs['HERGAST'])
    NMI = normalized_mutual_info_score(adata.obs['cell_type'], adata.obs['HERGAST'])
    FMI = fowlkes_mallows_score(adata.obs['cell_type'], adata.obs['HERGAST'])
    HC = homogeneity_score(adata.obs['cell_type'], adata.obs['HERGAST'])
    print(f'PCA_dim = {pc} | n_split = {split} | ARI = {round(ARI,3)} |  NMI = {round(NMI,3)} | FMI = {round(FMI,3)} | HC = {round(HC,3)} ')
