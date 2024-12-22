import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import pearsonr
import seaborn as sns
import HERGAST
plt.rcParams["figure.figsize"] = (9, 8)

# helper functions
def plot_gene_dist(adata,i,layer=None,num_bin=30,save=None):

    if layer is not None:
        df = pd.DataFrame({'value':adata.layers[layer][:,i],'mark':adata.obs['mark_area'].to_list()})
    else:
        df = pd.DataFrame({'value':adata.X[:,i],'mark':adata.obs['mark_area'].to_list()})
    
    # 设置绘图风格
    sns.set_theme(style="white")
    palette = {
    '0': '#6384BA',
    '1': '#E39C73'
    }

    plt.figure(figsize=(9, 6))
    sns.histplot(data=df, x='value', hue='mark', bins=num_bin, alpha=0.5, element='step',common_norm=False, stat='density',palette=palette)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.legend().set_visible(False)

    # 隐藏 x 和 y 轴标题
    plt.xlabel('')
    plt.ylabel('')

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def min_max_norm(data):
    # Min-Max 标准化
    min_vals = data.min(axis=0)  # 每列的最小值
    max_vals = data.max(axis=0)  # 每列的最大值

    # 执行标准化
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


dist_mapping={'0':'ZIP','1':'ZINB'}
dist = dist_mapping[sys.argv[1]]
device = int(sys.argv[2])
patterns = [f for f in os.listdir('.') if f.endswith('.png')]

for ptn in patterns:
    res_dir = ptn.split('.')[0]
    print(f'================================== pattern:{res_dir}  ===================================')
    res_dir = f'{dist}/{res_dir}'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if dist=='ZINB':
        adata = HERGAST.utils.simulate_gene(spots=160000,type='ZINB',ptn=ptn,se_mu=20,se_size=20,ns_mu=2,ns_size=2,se_p=0.01,ns_p=0.01)
        adata.layers['gt'] = adata.X
        # 设置高斯噪声的参数
        mean = 20  # 噪声均值
        stddev = 20  # 噪声标准差
        # 生成高斯噪声
        noise = np.random.normal(mean, stddev, adata.shape)
        # 将噪声添加到原始表达数据上
        adata.X = adata.layers['gt'] + noise
        adata.X = adata.X - adata.X.min()
    elif dist=='ZIP':
        adata = HERGAST.utils.simulate_gene(spots=160000,type='ZIP',ptn=ptn,se_mu=15,ns_mu=3,se_p=0.01,ns_p=0.01)
        adata.layers['gt'] = adata.X
        mean = 15  
        stddev = 15 
        noise = np.random.normal(mean, stddev, adata.shape)
        adata.X = adata.layers['gt'] + noise
        adata.X = adata.X - adata.X.min()
    adata.layers['noise'] = adata.X

    # HERGAST pipeline
    HERGAST.utils.Cal_Spatial_Net(adata,verbose=False)
    HERGAST.utils.Cal_Expression_Net(adata,verbose=False)

    train_HERGAST = HERGAST.Train_HERGAST(adata, batch_data=True, num_batch_x_y=(3,3), spatial_net_arg={'verbose':False},
                                        exp_net_arg={'verbose':False},dim_reduction=None,device_idx=device)
    train_HERGAST.train_HERGAST(n_epochs=200,save_reconstrction=True)

    #evaluation
    adata.layers['rex'] = adata.obsm['HERGAST_ReX']
    adata.layers['gt_norm'] = min_max_norm(adata.layers['gt'])
    adata.layers['rex_norm'] = min_max_norm(adata.layers['rex'])
    adata.layers['noise_norm'] = min_max_norm(adata.layers['noise'])
    rex_r = pearsonr(adata.layers['gt'][:,0], adata.layers['rex'][:,0])[0]
    noise_r = pearsonr(adata.layers['gt'][:,0], adata.layers['noise'][:,0])[0]
    print(f'rex: {rex_r} | noise: {noise_r}')


    # plotting
    sc.pl.embedding(adata,basis='spatial',color=f'se.1',cmap='Reds',layer='rex',show=False)
    plt.savefig(f'{res_dir}/svg_sp_rex.png')
    sc.pl.embedding(adata,basis='spatial',color=f'se.1',cmap='Reds',layer='gt',show=False)
    plt.savefig(f'{res_dir}/svg_sp_gt.png')
    sc.pl.embedding(adata,basis='spatial',color=f'se.1',cmap='Reds',layer='noise',show=False)
    plt.savefig(f'{res_dir}/svg_sp_noise.png')

    plot_gene_dist(adata,0,num_bin=25,layer='gt_norm',save=f'{res_dir}/svg_dist_gt.svg')
    plot_gene_dist(adata,0,num_bin=35,layer='rex_norm',save=f'{res_dir}/svg_dist_rex.svg')
    plot_gene_dist(adata,0,num_bin=35,layer='noise_norm',save=f'{res_dir}/svg_dist_noise.svg')


    # negative control
    # rex_r = pearsonr(adata.layers['gt'][:,50], adata.layers['rex'][:,50])[0]
    # noise_r = pearsonr(adata.layers['gt'][:,50], adata.layers['noise'][:,50])[0]
    # print(f'negativa control | rex: {rex_r} | noise: {noise_r}')

    # sc.pl.embedding(adata,basis='spatial',color='ns.1',cmap='Reds',layer='rex',show=False)
    # plt.savefig(f'{res_dir}/nsvg_sp_rex.png')
    # sc.pl.embedding(adata,basis='spatial',color='ns.1',cmap='Reds',layer='gt',show=False)
    # plt.savefig(f'{res_dir}/nsvg_sp_gt.png')
    # sc.pl.embedding(adata,basis='spatial',color='ns.1',cmap='Reds',layer='noise',show=False)
    # plt.savefig(f'{res_dir}/nsvg_sp_noise.png')

    # plot_gene_dist(adata,50,num_bin=25,layer='gt_norm',save=f'{res_dir}/nsvg_dist_gt.svg')
    # plot_gene_dist(adata,50,num_bin=35,layer='rex_norm',save=f'{res_dir}/nsvg_dist_rex.svg')
    # plot_gene_dist(adata,50,num_bin=35,layer='noise_norm',save=f'{res_dir}/nsvg_dist_noise.svg')