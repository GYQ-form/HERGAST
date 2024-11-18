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
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import contextlib
import io
import warnings

def Transfer_pytorch_Data(adata, dim_reduction=None, center_msg='out'):
    """
    Construct graph data for training.

    Parameters
    ----------
    adata : AnnData
        AnnData object which contains Spatial network and Expression network.
    dim_reduction : str or None
        Dimensional reduction methods (or the input feature). Can be 'PCA', 
        'HVG' or None (default uses all gene expression, which may cause out of memory during training).
    center_msg : str
        Message passing mode through the graph. Given a center spot, 
        'in' denotes that the message is flowing from connected spots to the center spot,
        'out' denotes that the message is flowing from the center spot to the connected spots.

    Returns
    -------
    data : Data
        The constructed graph data containing edges and features for training.
    """
    
    # Expression edge construction
    G_df = adata.uns['Exp_Net'].copy()  # Copy the expression network DataFrame
    cells = np.array(adata.obs_names)  # Get cell names from AnnData object
    # Create a mapping from cell names to indices
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # Map cell names to their corresponding indices
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # Create a sparse matrix for the expression edges
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # Add self-loops to the graph
    exp_edge = np.nonzero(G)  # Get the non-zero indices of the expression graph

    # Spatial edge construction
    G_df = adata.uns['Spatial_Net'].copy()  # Copy the spatial network DataFrame
    cells = np.array(adata.obs_names)  # Get cell names from AnnData object
    # Create a mapping from cell names to indices
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # Map cell names to their corresponding indices
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # Create a sparse matrix for the spatial edges
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # Add self-loops to the graph
    spatial_edge = np.nonzero(G)  # Get the non-zero indices of the spatial graph

    # Feature extraction based on the specified dimensional reduction method
    if dim_reduction == 'PCA':
        feat = adata.obsm['X_pca']  # Use PCA-reduced features
    elif dim_reduction == 'HVG':
        # Filter AnnData for highly variable genes
        adata_Vars = adata[:, adata.var['highly_variable']]
        # Convert sparse matrix to dense array if necessary
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()  # Convert to dense array
        else:
            feat = adata_Vars.X
    else:
        # Use all gene expression data if no dimensional reduction is specified
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat = adata.X.toarray()  # Convert to dense array
        else:
            feat = adata.X  # Use the existing dense array

    # Construct the graph data based on the message passing mode
    if center_msg == 'out':
        # Create edge indices for outgoing messages
        data = Data(edge_index=torch.LongTensor(np.array(
            [np.concatenate((exp_edge[0], spatial_edge[0])),
            np.concatenate((exp_edge[1], spatial_edge[1]))])).contiguous(), 
            x=torch.FloatTensor(feat.copy()))  # Use feature tensor
    else:
        # Create edge indices for incoming messages
        data = Data(edge_index=torch.LongTensor(np.array(
            [np.concatenate((exp_edge[1], spatial_edge[1])),
            np.concatenate((exp_edge[0], spatial_edge[0]))])).contiguous(), 
            x=torch.FloatTensor(feat.copy()))

    # Create edge types for the constructed graph
    edge_type = torch.zeros(exp_edge[0].shape[0] + spatial_edge[0].shape[0], dtype=torch.int64)
    edge_type[exp_edge[0].shape[0]:] = 1  # Set spatial edges to type 1
    data.edge_type = edge_type  # Assign edge types to the data

    return data  # Return the constructed graph data

# def Transfer_pytorch_Data(adata, dim_reduction=None, center_msg='out'):

#     #Expression edge
#     G_df = adata.uns['Exp_Net'].copy()
#     cells = np.array(adata.obs_names)
#     cells_id_tran = dict(zip(cells, range(cells.shape[0])))
#     G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
#     G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
#     G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
#     G = G + sp.eye(G.shape[0])
#     exp_edge = np.nonzero(G)

#     #Spatial edge
#     G_df = adata.uns['Spatial_Net'].copy()
#     cells = np.array(adata.obs_names)
#     cells_id_tran = dict(zip(cells, range(cells.shape[0])))
#     G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
#     G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
#     G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
#     G = G + sp.eye(G.shape[0])
#     spatial_edge = np.nonzero(G)
    
#     if dim_reduction=='PCA':
#         feat = adata.obsm['X_pca']
#     elif dim_reduction=='HVG':
#         adata_Vars = adata[:, adata.var['highly_variable']]
#         if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
#             feat = adata_Vars.X.toarray()
#         else:
#             feat = adata_Vars.X
#     else:
#         if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
#             feat = adata.X.toarray()
#         else:
#             feat = adata.X

#     if center_msg=='out':
#         data = Data(edge_index=torch.LongTensor(np.array(
#             [np.concatenate((exp_edge[0],spatial_edge[0])),
#             np.concatenate((exp_edge[1],spatial_edge[1]))])).contiguous(), 
#             x=torch.FloatTensor(feat.copy()))  # .todense()
#     else:
#         data = Data(edge_index=torch.LongTensor(np.array(
#             [np.concatenate((exp_edge[1],spatial_edge[1])),
#             np.concatenate((exp_edge[0],spatial_edge[0]))])).contiguous(), 
#             x=torch.FloatTensor(feat.copy()))
#     edge_type = torch.zeros(exp_edge[0].shape[0]+spatial_edge[0].shape[0],dtype=torch.int64)
#     edge_type[exp_edge[0].shape[0]:] = 1
#     data.edge_type = edge_type
        
#     return data

def Batch_Data(adata, num_batch_x, num_batch_y, plot_Stats=False):
    """
    Create batches of spatial data based on specified coordinates.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial information.
    num_batch_x : int
        Number of batches along the x-axis.
    num_batch_y : int
        Number of batches along the y-axis.
    plot_Stats : bool, optional
        If True, plot statistics of the number of spots in each batch.

    Returns
    -------
    Batch_list : list
        A list containing AnnData objects for each batch.
    """

    # Retrieve spatial coordinates from the AnnData object
    Sp_df = adata.obsm['spatial']

    # Calculate the x-coordinates for the specified number of batches
    batch_x_coor = np.percentile(Sp_df[:, 0], np.linspace(0, 100, num_batch_x + 1))
    # Calculate the y-coordinates for the specified number of batches
    batch_y_coor = np.percentile(Sp_df[:, 1], np.linspace(0, 100, num_batch_y + 1))

    # Initialize an empty list to store each batch of data
    Batch_list = []
    
    # Iterate over the number of batches along the x-axis
    for it_x in range(num_batch_x):
        # Get the min and max x-coordinates for the current batch
        min_x, max_x = batch_x_coor[it_x], batch_x_coor[it_x + 1]
        
        # Iterate over the number of batches along the y-axis
        for it_y in range(num_batch_y):
            # Get the min and max y-coordinates for the current batch
            min_y, max_y = batch_y_coor[it_y], batch_y_coor[it_y + 1]

            # Create a mask for the x-coordinate to filter the data
            mask_x = (Sp_df[:, 0] >= min_x) & (Sp_df[:, 0] <= max_x)
            # Create a mask for the y-coordinate to filter the data
            mask_y = (Sp_df[:, 1] >= min_y) & (Sp_df[:, 1] <= max_y)
            # Combine both masks to get the final mask
            mask = mask_x & mask_y

            # Create a temporary AnnData object for the current batch based on the mask
            temp_adata = adata[mask].copy()
            # Check if the batch contains more than 10 spots
            if temp_adata.shape[0] > 10:
                Batch_list.append(temp_adata)  # Add the valid batch to the list
            
    # If plot_Stats is True, visualize the distribution of spots per batch
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))  # Create a subplot for the boxplot
        # Create a DataFrame to hold the number of spots in each batch
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        # Create a boxplot to show the distribution of spots per batch
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        # Overlay a stripplot to show individual batch sizes
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)

    return Batch_list  # Return the list of batches

# def Batch_Data(adata, num_batch_x, num_batch_y, plot_Stats=False):

#     Sp_df = adata.obsm['spatial']

#     batch_x_coor = np.percentile(Sp_df[:, 0], np.linspace(0, 100, num_batch_x + 1))
#     batch_y_coor = np.percentile(Sp_df[:, 1], np.linspace(0, 100, num_batch_y + 1))

#     Batch_list = []
#     for it_x in range(num_batch_x):
#         min_x, max_x = batch_x_coor[it_x], batch_x_coor[it_x + 1]
#         for it_y in range(num_batch_y):
#             min_y, max_y = batch_y_coor[it_y], batch_y_coor[it_y + 1]

#             mask_x = (Sp_df[:, 0] >= min_x) & (Sp_df[:, 0] <= max_x)
#             mask_y = (Sp_df[:, 1] >= min_y) & (Sp_df[:, 1] <= max_y)
#             mask = mask_x & mask_y

#             temp_adata = adata[mask].copy()
#             if temp_adata.shape[0] > 10:
#                 Batch_list.append(temp_adata)
            
#     if plot_Stats:
#         f, ax = plt.subplots(figsize=(1, 3))
#         plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
#         sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
#         sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
#     return Batch_list


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=4, model='KNN', verbose=True):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata : AnnData
        AnnData object from the scanpy package containing spatial coordinates.
    rad_cutoff : float or None
        Radius cutoff used when model='Radius' to determine connectivity based on distance.
    k_cutoff : int
        The number of nearest neighbors when model='KNN'. 
        This parameter affects resource usage (time and memory). Reducing this can make training more efficient, but may slightly impact performance.
    model : str
        The network construction model. 
        When model=='Radius', spots are connected to others within the specified radius. 
        When model=='KNN', each spot is connected to its first k_cutoff nearest neighbors.
    
    Returns
    -------
    None
        The spatial networks are saved in adata.uns['Spatial_Net'].
    """

    # Ensure the specified model is valid
    assert model in ['Radius', 'KNN']
    
    # Print progress message if verbose mode is enabled
    if verbose:
        print('------Calculating spatial graph...')
    
    # Retrieve spatial coordinates from the AnnData object
    coor = adata.obsm['spatial']
    num_cells = coor.shape[0]  # Get the number of cells

    # Construct the neighbor graph based on the chosen model
    if model == 'Radius':
        # Use radius-based neighbor search
        nbrs = NearestNeighbors(radius=rad_cutoff).fit(coor)  # Fit the model with spatial coordinates
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)  # Get neighbors within the radius
    elif model == 'KNN':
        # Use k-nearest neighbors search
        nbrs = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)  # Fit the model with spatial coordinates
        distances, indices = nbrs.kneighbors(coor)  # Get the k nearest neighbors

    # Build the list of edges for the spatial network
    KNN_list = [
        (i, indices[i][j], distances[i][j])  # Create tuples of (cell index, neighbor index, distance)
        for i in range(num_cells)  # Iterate over each cell
        for j in range(len(indices[i]))  # Iterate over each neighbor for the current cell
        if distances[i][j] > 0  # Ensure that the distance is greater than zero (exclude self-loops)
    ]

    # Create a DataFrame from the list of edges
    KNN_df = pd.DataFrame(KNN_list, columns=['Cell1', 'Cell2', 'Distance'])

    # Map the indices back to the actual cell names
    id_cell_trans = np.array(adata.obs.index)  # Get the cell names from the AnnData object
    KNN_df['Cell1'] = id_cell_trans[KNN_df['Cell1']]  # Map Cell1 indices to names
    KNN_df['Cell2'] = id_cell_trans[KNN_df['Cell2']]  # Map Cell2 indices to names

    # Print summary statistics if verbose mode is enabled
    if verbose:
        print(f'Spatial graph contains {KNN_df.shape[0]} edges, {adata.n_obs} cells.')  # Number of edges and cells
        print(f'{KNN_df.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')  # Average number of neighbors per cell

    # Save the spatial network DataFrame into the AnnData object
    adata.uns['Spatial_Net'] = KNN_df  # Store the constructed spatial network

# def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=4, model='KNN', verbose=True):
#     """
#     Construct the spatial neighbor networks.

#     Parameters
#     ----------
#     adata
#         AnnData object of scanpy package.
#     rad_cutoff
#         radius cutoff when model='Radius'
#     k_cutoff
#         The number of nearest neighbors when model='KNN'.
#         This parameter determines the maximun resource usage (time and memory), you can reduce the parameter for more efficient training, but the model's performance may be slightly influenced.
#     model
#         The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    
#     Returns
#     -------
#     The spatial networks are saved in adata.uns['Spatial_Net']
#     """

#     assert model in ['Radius', 'KNN']
#     if verbose:
#         print('------Calculating spatial graph...')
    
#     coor = adata.obsm['spatial']
#     num_cells = coor.shape[0]

#     if model == 'Radius':
#         nbrs = NearestNeighbors(radius=rad_cutoff).fit(coor)
#         distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
#     elif model == 'KNN':
#         nbrs = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
#         distances, indices = nbrs.kneighbors(coor)

#     # Build the network
#     KNN_list = [
#         (i, indices[i][j], distances[i][j])
#         for i in range(num_cells)
#         for j in range(len(indices[i]))
#         if distances[i][j] > 0
#     ]

#     KNN_df = pd.DataFrame(KNN_list, columns=['Cell1', 'Cell2', 'Distance'])

#     # Map indices to cell names
#     id_cell_trans = np.array(adata.obs.index)
#     KNN_df['Cell1'] = id_cell_trans[KNN_df['Cell1']]
#     KNN_df['Cell2'] = id_cell_trans[KNN_df['Cell2']]

#     if verbose:
#         print(f'Spatial graph contains {KNN_df.shape[0]} edges, {adata.n_obs} cells.')
#         print(f'{KNN_df.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')

#     adata.uns['Spatial_Net'] = KNN_df

def Cal_Expression_Net(adata, k_cutoff=3, dim_reduce=None, verbose=True):
    """
    Construct an expression similarity network based on gene expression data.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression data.
    k_cutoff : int
        The number of nearest neighbors to consider for each cell.
    dim_reduce : str or None
        Dimensional reduction method to apply. Options are 'PCA', 'HVG', or None.
    verbose : bool
        If True, print progress messages and summary statistics.

    Returns
    -------
    None
        The expression similarity network is saved in adata.uns['Exp_Net'].
    """

    # Print progress message if verbose mode is enabled
    if verbose:
        print('------Calculating Expression similarity graph...')

    # Select feature representation based on the specified dimensional reduction method
    if dim_reduce == 'PCA':
        coor = adata.obsm['X_pca']  # Use PCA-reduced coordinates
    elif dim_reduce == 'HVG':
        # Filter for highly variable genes
        adata_Vars = adata[:, adata.var['highly_variable']]
        # Convert sparse matrix to dense array if necessary
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()  # Convert to dense array
        else:
            feat = adata_Vars.X  # Use the existing dense array
        # Create a DataFrame for the features of highly variable genes
        coor = pd.DataFrame(feat)
        coor.index = adata.obs.index  # Set the index to the cell names
        coor.columns = adata.var_names[adata.var['highly_variable']]  # Set columns to variable gene names
        adata.obsm['HVG'] = coor  # Store this in the AnnData object
    else:
        # Warn that no dimensional reduction method was specified and use all genes
        warnings.warn("No dimensional reduction method specified, using all genes' expression to calculate expression similarity network.")
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat = adata.X.toarray()  # Convert to dense array if necessary
        else:
            feat = adata.X  # Use the existing dense array
        coor = feat  # Set coordinates to the full feature set

    # Determine the number of neighbors to find, ensuring it does not exceed the number of cells
    n_nbrs = k_cutoff + 1 if k_cutoff + 1 < coor.shape[0] else coor.shape[0]
    # Fit the nearest neighbors model to the coordinates
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_nbrs).fit(coor)
    # Retrieve distances and indices of the nearest neighbors
    distances, indices = nbrs.kneighbors(coor)

    # Build a list of edges for the expression network
    KNN_list = [
        (i, indices[i][j], distances[i][j])  # Create tuples of (cell index, neighbor index, distance)
        for i in range(coor.shape[0])  # Iterate over each cell
        for j in range(len(indices[i]))  # Iterate over each neighbor for the current cell
        if distances[i][j] > 0  # Exclude self-loops (distance must be greater than zero)
    ]

    # Create a DataFrame from the list of edges
    KNN_df = pd.DataFrame(KNN_list, columns=['Cell1', 'Cell2', 'Distance'])

    # Map the indices back to actual cell names
    id_cell_trans = np.array(adata.obs.index)  # Get the cell names from the AnnData object
    KNN_df['Cell1'] = id_cell_trans[KNN_df['Cell1']]  # Map Cell1 indices to names
    KNN_df['Cell2'] = id_cell_trans[KNN_df['Cell2']]  # Map Cell2 indices to names

    # Print summary statistics if verbose mode is enabled
    if verbose:
        print(f'Expression graph contains {KNN_df.shape[0]} edges, {adata.n_obs} cells.')  # Number of edges and cells
        print(f'{KNN_df.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')  # Average number of neighbors per cell

    # Save the expression network DataFrame into the AnnData object
    adata.uns['Exp_Net'] = KNN_df  # Store the constructed expression network

# def Cal_Expression_Net(adata, k_cutoff=3, dim_reduce=None, verbose=True):

#     if verbose:
#         print('------Calculating Expression simalarity graph...')

#     if dim_reduce=='PCA':
#         coor = adata.obsm['X_pca']
#     elif dim_reduce=='HVG':
#         adata_Vars = adata[:, adata.var['highly_variable']]
#         if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
#             feat = adata_Vars.X.toarray()
#         else:
#             feat = adata_Vars.X
#         coor = pd.DataFrame(feat)
#         coor.index = adata.obs.index
#         coor.columns = adata.var_names[adata.var['highly_variable']]
#         adata.obsm['HVG'] = coor
#     else:
#         warnings.warn("No dimentional reduction method specified, using all genes' expression to calculate expression similarity network.")
#         if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
#             feat = adata.X.toarray()
#         else:
#             feat = adata.X
#         coor = feat

#     n_nbrs = k_cutoff+1 if k_cutoff+1<coor.shape[0] else coor.shape[0]
#     nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_nbrs).fit(coor)
#     distances, indices = nbrs.kneighbors(coor)
#     KNN_list = [
#         (i, indices[i][j], distances[i][j])
#         for i in range(coor.shape[0])
#         for j in range(len(indices[i]))
#         if distances[i][j] > 0
#     ]

#     KNN_df = pd.DataFrame(KNN_list, columns=['Cell1', 'Cell2', 'Distance'])

#     # Map indices to cell names
#     id_cell_trans = np.array(adata.obs.index)
#     KNN_df['Cell1'] = id_cell_trans[KNN_df['Cell1']]
#     KNN_df['Cell2'] = id_cell_trans[KNN_df['Cell2']]

#     if verbose:
#         print(f'Expression graph contains {KNN_df.shape[0]} edges, {adata.n_obs} cells.')
#         print(f'{KNN_df.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')

#     adata.uns['Exp_Net'] = KNN_df
    
def cal_metagene(adata, gene_list, obs_name='metagene', layer=None):
    """
    Calculate the metagene expression for a specified list of genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression data.
    gene_list : list of str
        List of gene names for which to calculate the metagene.
    obs_name : str
        Name of the observation to store the metagene expression in adata.obs.
    layer : str or None
        Optional layer from which to extract gene expressions. If None, uses the main expression matrix.

    Returns
    -------
    None
        The metagene expression is saved in adata.obs under the specified obs_name.
    """

    # Check if a specific layer is provided for gene expressions
    if layer is not None:
        # Extract gene expressions from the specified layer
        gene_expressions = adata[:, gene_list].layers[layer]
    else:
        # Extract gene expressions from the main expression matrix
        gene_expressions = adata[:, gene_list].X

    # Check if the gene expressions are in sparse format
    if sp.issparse(gene_expressions):
        # Convert sparse matrix to a dense array for easier manipulation
        gene_expressions = gene_expressions.toarray()

    # Calculate the metagene expression by summing the expressions of the specified genes across cells
    metagene_expression = np.sum(gene_expressions, axis=1)

    # Store the calculated metagene expression in the AnnData object's observations
    adata.obs[obs_name] = metagene_expression

# def cal_metagene(adata,gene_list,obs_name='metagene',layer=None):

#     if layer is not None:
#         gene_expressions = adata[:, gene_list].layers[layer]
#     else:
#         gene_expressions = adata[:, gene_list].X

#     if sp.issparse(gene_expressions):
#         gene_expressions = gene_expressions.toarray()

#     metagene_expression = np.sum(gene_expressions, axis=1)

#     adata.obs[obs_name] = metagene_expression


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
