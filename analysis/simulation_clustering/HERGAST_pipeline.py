import time
import torch
from memory_profiler import memory_usage
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
import math
import HERGAST
plt.rcParams["figure.figsize"] = (10, 10)

simu = sys.argv[1]
rep = sys.argv[2]
device_idx = sys.argv[3]

def combine_cluster(adata,key):
    cat_counts = adata.obs[key].value_counts()
    small_cats = cat_counts[cat_counts < adata.shape[0] / 10000].index
    adata.obs[key] = adata.obs[key].astype('str')
    adata.obs.loc[adata.obs[key].isin(small_cats), key] = 'Others'

def measure_resources(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()

        mem_usage = memory_usage((func, args, kwargs), max_usage=True) / 1024

        end_time = time.time()
        execution_time = end_time - start_time

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device_idx}")
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3) 
            cached = torch.cuda.memory_reserved(device) / (1024 ** 3)
        else:
            allocated = cached = 0

        print(f"Function '{func.__name__}' executed in {execution_time/60:.4f} minutes.")
        print(f"Memory usage: {mem_usage:.2f} GB")
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory cached: {cached:.2f} GB")

    return wrapper

print(f'##############################      rep:{rep}       ############################')

res_dir = f"simu{simu}/rep{rep}"
adata = sc.read_h5ad(f'{res_dir}/data.h5ad')
split = max(round(np.sqrt(adata.shape[0]/10000)),1)

# preprocess
@measure_resources
def preprocess(adata):
    sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
    sc.pp.scale(adata)
    sc.pp.pca(adata, n_comps=200)
preprocess(adata)


####### HERGAST
@measure_resources
def construct_net(adata):
    HERGAST.utils.Cal_Spatial_Net(adata,verbose=False)
    HERGAST.utils.Cal_Expression_Net(adata, dim_reduce='PCA',verbose=False)
    global train_HERGAST
    train_HERGAST = HERGAST.Train_HERGAST(adata, batch_data=True if split > 1 else False, num_batch_x_y=(split,split),dim_reduction='PCA',spatial_net_arg={'verbose':False},
                                device_idx = device_idx, exp_net_arg={'verbose':False})
construct_net(adata)


@measure_resources
def HERGAST_training(train_HERGAST):
    train_HERGAST.train_HERGAST(save_path=None, n_epochs=200,save_loss=False,save_reconstrction=False)
HERGAST_training(train_HERGAST)

@measure_resources
def post_process(adata):
    sc.pp.neighbors(adata, use_rep='HERGAST')
    used_res = 0.01
    sc.tl.leiden(adata, random_state=2024, resolution=used_res,key_added='HERGAST')
    combine_cluster(adata,'HERGAST')
post_process(adata)

ARI = adjusted_rand_score(adata.obs['ann_level_3'], adata.obs['HERGAST'])
NMI = normalized_mutual_info_score(adata.obs['ann_level_3'], adata.obs['HERGAST'])
FMI = fowlkes_mallows_score(adata.obs['ann_level_3'], adata.obs['HERGAST'])
HC = homogeneity_score(adata.obs['ann_level_3'], adata.obs['HERGAST'])
print(f'HERGAST | ARI = {round(ARI,3)} |  NMI = {round(NMI,3)} | FMI = {round(FMI,3)} | HC = {round(HC,3)} ')

sc.pl.embedding(adata, show=False, color='HERGAST',basis='spatial',s=6,title=f'HERGAST:{round(ARI,3)}',palette='tab20')
plt.savefig(f'{res_dir}/HERGAST.pdf', bbox_inches='tight')
