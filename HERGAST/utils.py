import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import scanpy as sc
import torch
from torch_geometric.data import Data
import contextlib
import io
import warnings

def Transfer_pytorch_Data(adata, dim_reduction=None, center_msg='out'):
    """\
    Construct graph data for training.

    Parameters
    ----------
    adata
        AnnData object which contains Spatial network and Expression network.
    dim_reduction
        Dimensional reduction methods (or the input feature). Can be 'PCA', 
        'HVG' or None (default using all gene expression, which may cause out of memeory when training).
    center_msg
        Message passing mode through the graph. Given a center spot, 
        'in' denotes that the message is flowing from connected spots to the center spot,
        'out' denotes that the message is flowing from the center spot to the connected spots.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    #Expression edge
    G_df = adata.uns['Exp_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    exp_edge = np.nonzero(G)

    #Spatial edge
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    spatial_edge = np.nonzero(G)
    
    if dim_reduction=='PCA':
        feat = adata.obsm['X_pca']
    elif dim_reduction=='HVG':
        adata_Vars = adata[:, adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()
        else:
            feat = adata_Vars.X
    else:
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat = adata.X.toarray()
        else:
            feat = adata.X

    if center_msg=='out':
        data = Data(edge_index=torch.LongTensor(np.array(
            [np.concatenate((exp_edge[0],spatial_edge[0])),
            np.concatenate((exp_edge[1],spatial_edge[1]))])).contiguous(), 
            x=torch.FloatTensor(feat.copy()))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [np.concatenate((exp_edge[1],spatial_edge[1])),
            np.concatenate((exp_edge[0],spatial_edge[0]))])).contiguous(), 
            x=torch.FloatTensor(feat.copy()))
    edge_type = torch.zeros(exp_edge[0].shape[0]+spatial_edge[0].shape[0],dtype=torch.int64)
    edge_type[exp_edge[0].shape[0]:] = 1
    data.edge_type = edge_type
        
    return data

def Batch_Data(adata, num_batch_x, num_batch_y, plot_Stats=False):
    # 提取所需的空间坐标数据并转换为 numpy 数组
    Sp_df = adata.obsm['spatial']

    # 计算分批的坐标范围
    batch_x_coor = np.percentile(Sp_df[:, 0], np.linspace(0, 100, num_batch_x + 1))
    batch_y_coor = np.percentile(Sp_df[:, 1], np.linspace(0, 100, num_batch_y + 1))

    Batch_list = []
    for it_x in range(num_batch_x):
        min_x, max_x = batch_x_coor[it_x], batch_x_coor[it_x + 1]
        for it_y in range(num_batch_y):
            min_y, max_y = batch_y_coor[it_y], batch_y_coor[it_y + 1]

            # 使用布尔索引进行空间坐标过滤
            mask_x = (Sp_df[:, 0] >= min_x) & (Sp_df[:, 0] <= max_x)
            mask_y = (Sp_df[:, 1] >= min_y) & (Sp_df[:, 1] <= max_y)
            mask = mask_x & mask_y

            # 生成子集并添加到列表中
            temp_adata = adata[mask].copy()
            if temp_adata.shape[0] > 10:
                Batch_list.append(temp_adata)
            
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=8, model='KNN', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('Spatial graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net

def Cal_Expression_Net(adata, k_cutoff=6, dim_reduce=None, verbose=True):

    if verbose:
        print('------Calculating Expression simalarity graph...')

    if dim_reduce=='PCA':
        coor = pd.DataFrame(adata.obsm['X_pca'])
        coor.index = adata.obs.index
    elif dim_reduce=='HVG':
        adata_Vars = adata[:, adata.var['highly_variable']]
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()
        else:
            feat = adata_Vars.X
        coor = pd.DataFrame(feat)
        coor.index = adata.obs.index
        coor.columns = adata.var_names[adata.var['highly_variable']]
        adata.obsm['HVG'] = coor
    else:
        warnings.warn("No dimentional reduction method specified, using all genes' expression to calculate expression similarity network.")
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat = adata.X.toarray()
        else:
            feat = adata.X
        coor = pd.DataFrame(feat)
        coor.index = adata.obs.index
        coor.columns = adata.var_names

    n_nbrs = k_cutoff+1 if k_cutoff+1<coor.shape[0] else coor.shape[0]
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_nbrs).fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    exp_Net = KNN_df.copy()
    exp_Net = exp_Net.loc[exp_Net['Distance']>0,]

    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    exp_Net['Cell1'] = exp_Net['Cell1'].map(id_cell_trans)
    exp_Net['Cell2'] = exp_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('Expression graph contains %d edges, %d cells.' %(exp_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(exp_Net.shape[0]/adata.n_obs))

    adata.uns['Exp_Net'] = exp_Net
    

def cal_metagene(adata,gene_list,obs_name='metagene',layer=None):

    # 提取感兴趣基因的表达矩阵
    if layer is not None:
        gene_expressions = adata[:, gene_list].layers[layer]
    else:
        gene_expressions = adata[:, gene_list].X

    # 如果是稀疏矩阵，则转换为密集矩阵
    if sp.issparse(gene_expressions):
        gene_expressions = gene_expressions.toarray()

    # 计算给定基因列表的表达值之和
    metagene_expression = np.sum(gene_expressions, axis=1)

    # 将新的 metagene 添加到 anndata 对象中
    adata.obs[obs_name] = metagene_expression


def simulate_ST(sc_adata, spatial_df, sc_type_col='ann_level_3', sp_type_col='domain', disperse_frac=0.3):
    """
    Simulate spatial transcriptomics data from single-cell RNA-seq data.
    
    Parameters:
    - sc_adata: AnnData object containing single-cell RNA-seq data.
    - spatial_df: DataFrame containing spatial coordinates and cell type labels.
    
    Returns:
    - AnnData object with spatial transcriptomics data. Spatial coordinates are in obsm['spatial'] and cell types are in obs.
    """
    
    # Get unique cell types from single-cell data
    unique_sc_types = sc_adata.obs[sc_type_col].unique()
    
    # Get unique cell types from spatial data
    unique_spatial_types = spatial_df[sp_type_col].unique()
    
    # Randomly select cell types for each spatial type
    selected_types = np.random.choice(unique_sc_types, len(unique_spatial_types), replace=False)
    
    # Create a mapping from spatial types to selected single-cell types
    type_mapping = dict(zip(unique_spatial_types, selected_types))
    
    # Initialize a list to collect the simulated cell indices
    simulated_indices = []
    spatial_coords = []
    
    # Iterate over each cell type in the spatial data
    for spatial_type, count in spatial_df[sp_type_col].value_counts().items():
        # Get the corresponding single-cell type
        sc_type = type_mapping[spatial_type]
        print(f'sptial type:{spatial_type}, sc type:{sc_type}')
        
        # Get all cells of this type from the single-cell data
        sc_cells_of_type_indices = np.where(sc_adata.obs[sc_type_col] == sc_type)[0]
        
        if len(sc_cells_of_type_indices) >= count:
            # If we have enough cells, randomly select 'count' cells
            selected_indices = np.random.choice(sc_cells_of_type_indices, count, replace=False)
        else:
            # If not enough cells, randomly assign the available cells to the spatial positions
            selected_indices = np.random.choice(sc_cells_of_type_indices, count, replace=True)

        # Collect the selected cell indices
        simulated_indices.extend(selected_indices)
        
        # Collect the corresponding spatial coordinates
        spatial_coords.append(spatial_df[spatial_df[sp_type_col] == spatial_type][['x', 'y']].values)

    # Randomly select a cell type to simulate the dispersed cells
    remaining_types = list(set(unique_sc_types) - set(selected_types))
    if remaining_types:
        dispersed_type = np.random.choice(remaining_types)
        print(f'disperse cell type:{dispersed_type}')
        dispersed_cells_indices = np.where(sc_adata.obs[sc_type_col] == dispersed_type)[0]
        dispersed_sample_size = round(spatial_df.shape[0] * disperse_frac)
        if dispersed_cells_indices.shape[0] >= dispersed_sample_size:
            dispersed_cells_indices = np.random.choice(dispersed_cells_indices, dispersed_sample_size, replace=False)
        else:
            dispersed_cells_indices = np.random.choice(dispersed_cells_indices, dispersed_sample_size, replace=True)
        
        # Replace some cells in the simulated_adata with dispersed cells
        replace_indices = np.random.choice(len(simulated_indices), dispersed_sample_size, replace=False)
        simulated_indices_array = np.array(simulated_indices)
        simulated_indices_array[replace_indices] = dispersed_cells_indices

    # Concatenate all selected cells to form the simulated spatial data
    simulated_adata = sc_adata[simulated_indices_array].copy()
    simulated_adata.obs_names_make_unique()

    # Set the spatial coordinates
    simulated_adata.obsm['spatial'] = np.vstack(spatial_coords)
    simulated_adata.obs[sc_type_col] = simulated_adata.obs[sc_type_col].astype('str')
    simulated_adata.obs[sc_type_col].iloc[replace_indices] = dispersed_type
    
    return simulated_adata

def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)