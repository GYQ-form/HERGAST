import numpy as np
import random
from tqdm import tqdm
import warnings
from .HERGAST import HERGAST
from .utils import Transfer_Graph_Data, Batch_Data, Cal_Spatial_Net, Cal_Expression_Net

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


class Train_HERGAST:

    def __init__(self, adata, dim_reduction = 'PCA', batch_data = False, num_batch_x_y = None, device_idx = 0, spatial_net_arg = {}, exp_net_arg = {}, verbose=False, center_msg='out'):

        """
        Initialization of a HERGAST trainer.

        Parameters
        ----------
        adata : AnnData
            AnnData object from the anndata package.
        dim_reduction : str
            Dimensional reduction method used as input for HERGAST. Options are 'PCA' or 'HVG'. Default is 'PCA'.
            'HVG' is recommended for meaningful gene expression reconstruction.
            If set to None or an unrecognized method, all gene expressions will be used as input (recommended for small gene counts, e.g. Xenium platform).
        batch_data : bool
            Indicates whether to use the DIC strategy. Default is False.
            Set to False for traditional ST platforms (10X Visium, MERFISH, Slide-seq).
            Set to True for large-scale ST data (Visium HD, Xenium, etc).
        num_batch_x_y : tuple
            Specifies the number of points to segment the spatially transcribed slice on the x and y axes.
            Each segmented patch is trained iteratively. Relevant only when batch_data=True.
        device_idx : int
            Specifies which GPU to use (if multiple GPUs are available).
        spatial_net_arg : dict
            Arguments passed to calculate the spatial network in each batch. See `Cal_Spatial_Net`.
        exp_net_arg : dict
            Arguments passed to calculate the expression network in each batch. See `Cal_Expression_Net`.
        verbose : bool
            If True, prints detailed running information. Default is False.
        center_msg : str
            In the message passing GNN mechanism:
            'out' : messages flow out from the center spot to neighboring spots.
            'in' : messages flow into the center spot from neighboring spots.
        """

        # Validate the dimensional reduction method and ensure necessary computations are done
        if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")
        elif dim_reduction == 'HVG':
            if 'highly_variable' not in adata.var.keys():
                raise ValueError("HVG has not been computed! Run sc.pp.highly_variable_genes first!")
        else:
            # Warn that no valid dimensional reduction method was specified
            warnings.warn("No dimentional reduction method specified, using all genes' expression as input.")
        
        # Store parameters in instance variables
        self.dim_reduction = dim_reduction
        self.batch_data = batch_data
        self.adata = adata

        # Ensure that the spatial and expression networks exist in the AnnData object
        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        if 'Exp_Net' not in adata.uns.keys():
            raise ValueError("Exp_Net is not existed! Run Cal_Expression_Net first!")
        
        # Prepare the graph data for training
        self.data = Transfer_Graph_Data(adata, dim_reduction=dim_reduction,center_msg=center_msg)
        if verbose:
            print('Size of Input: ', self.data.x.shape)

        # If DIC is enabled, create patches and compute sub-networks
        if batch_data:
            self.num_batch_x, self.num_batch_y = num_batch_x_y
            # Create batches of data based on the specified number of segments
            Batch_list = Batch_Data(adata, num_batch_x=self.num_batch_x, num_batch_y=self.num_batch_y)
            # For each batch of data, calculate spatial and expression networks
            for temp_adata in Batch_list:
                Cal_Spatial_Net(temp_adata, **spatial_net_arg)
                Cal_Expression_Net(temp_adata, dim_reduce=dim_reduction, **exp_net_arg)

            # Transfer each batch data to graph data
            data_list = [Transfer_Graph_Data(adata, dim_reduction=dim_reduction,center_msg=center_msg) for adata in Batch_list]
            # Create a DataLoader for the batches
            self.loader = DataLoader(data_list, batch_size=1, shuffle=True)

        # Set the device for training (GPU or CPU)
        self.device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
        self.model = None


    def train_HERGAST(self, save_path = None, hidden_dims=[100, 32], n_epochs=200, lr=0.001, 
                    key_added='HERGAST', att_drop = 0.3, gradient_clipping=5., weight_decay=0.0001,
                    random_seed=2024, save_loss=False, save_reconstrction=True):

        """
        Training graph attention auto-encoder.

        Parameters
        ----------
        save_path : str or None
            directory to save the trained HERGAST model. Default is None (not save the model).
        hidden_dims : list of int
            The dimension of the 1st and 2nd layer of HERGAST encoder.
        n_epochs : int
            Number of total epochs in training.
        lr : float
            Learning rate for AdamOptimizer.
        key_added : str
            The latent embeddings are saved in adata.obsm[key_added].
        att_drop : float
            Dropout probability of the normalized attention coefficients which exposes each node to a stochastically sampled neighborhood during training.
        gradient_clipping : float
            Gradient clipping threshold.
        weight_decay : float
            Weight decay for AdamOptimizer.
        random_seed : int
            Randomization seed.
        save_loss : bool
            If True, the training loss is saved in adata.uns['HERGAST_loss']. Defalut is False.
        save_reconstrction : bool
            If True, the reconstructed profiles are saved in adata.layers['HERGAST_ReX'].
            This is the reconstructed gene expression profile when the input is gene expression profile, but meaningless when input profile is PCA representation.
        """

        # Set the path where the model will be saved
        self.save_path = save_path

        # Set random seeds
        seed=random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Initialize the model
        if self.model is None:
            # Create a new HERGAST model if one does not exist
            model = HERGAST(hidden_dims = [self.data.x.shape[1]] + hidden_dims, dim_reduce=self.dim_reduction, att_drop=att_drop).to(self.device)
        else:
            # Use the existing model and move it to the specified device
            model = self.model.to(self.device)
        
        # Move data to the appropriate device (CPU or GPU)
        if self.batch_data:
            data = self.data.to('cpu') # Load data to CPU if using batch data
        else:
            data = self.data.to(self.device)  # Load data to the specified device

        # Set up the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

         # List to keep track of loss values during training
        loss_list = []

        with tqdm(range(n_epochs)) as tq:
            for _ in tq:
                if self.batch_data:
                    # Loop through batches of data if DIC is enabled
                    for batch in self.loader:
                        batch = batch.to(self.device)
                        model.train()
                        optimizer.zero_grad()
                        z, out = model(batch.x, batch.edge_index, batch.edge_type)
                        loss = F.mse_loss(batch.x, out)
                        loss.backward()
                        loss_list.append(loss.item())
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        optimizer.step()

                else:
                    # Training without DIC
                    model.train()
                    optimizer.zero_grad()
                    z, out = model(data.x, data.edge_index, data.edge_type)
                    loss = F.mse_loss(data.x, out)
                    loss.backward()
                    loss_list.append(loss.item())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()

        # Evaluation phase after training
        with torch.no_grad():
            if self.batch_data:
                # Move model to CPU for evaluation if using DIC
                model.to('cpu')
                model.eval()
                z, out = model(data.x.cpu(), data.edge_index.cpu(), data.edge_type.cpu())
                model.to(self.device)
            else:
                model.eval()
                z, out = model(data.x, data.edge_index, data.edge_type)

        # Store the HERGAST representations in the AnnData object
        HERGAST_rep = z.to('cpu').detach().numpy()
        self.adata.obsm[key_added] = HERGAST_rep

        # Save the trained model if a save path is provided
        if save_path is not None:
            torch.save(model,f'{save_path}/model.pth')

        # Save loss values if requested
        if save_loss:
            self.adata.uns['HERGAST_loss'] = loss_list

        # Save reconstructed output if requested
        if save_reconstrction:
            ReX = out.to('cpu').numpy()
            if self.dim_reduction != 'PCA':
                idx = np.where(self.adata.X==0)
                ReX[idx] = 0
            self.adata.obsm['HERGAST_ReX'] = ReX

        self.model = model

    def load_model(self, path):

        """
        Load a trained HERGAST model.

        Parameters
        ----------
        path
            Directory to load the trained HERGAST model
        """

        self.model = torch.load(path, map_location=self.device)

    def save_model(self, path):

        """
        Save a trained HERGAST model.

        Parameters
        ----------
        path
            Directory to save the trained HERGAST model
        """

        torch.save(self.model,f'{path}/model.pth')

    @torch.no_grad()
    def process(self, gdata = None):

        """
        Conduct inference using the HERGAST model.

        Parameters
        ----------
        gdata
            Graph data used for inference.
            If not specified, will use the registered whole slice ST data as input.
        """

        if gdata is None:
            gdata = self.data
        self.model.to('cpu')
        self.model.eval()
        gdata = gdata.to('cpu')
        z,out =  self.model(gdata.x, gdata.edge_index, gdata.edge_type)
        HERGAST_rep = z.to('cpu').detach().numpy()
        self.adata.obsm['HERGAST'] = HERGAST_rep
        if self.dim_reduction != 'PCA':
            ReX = out.to('cpu').detach().numpy()
            idx = np.where(self.adata.X==0)
            ReX[idx] = 0
            self.adata.obsm['HERGAST_ReX'] = ReX

