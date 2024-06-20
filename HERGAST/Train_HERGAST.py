import numpy as np
import random
from tqdm import tqdm
import warnings
from .HERGAST import HERGAST
from .utils import Transfer_pytorch_Data, Batch_Data, Cal_Spatial_Net, Cal_Expression_Net

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.loader import DataLoader


class Train_HERGAST:

    def __init__(self, adata, dim_reduction = None, batch_data = False, num_batch_x_y = None, device_idx = 7, spatial_net_arg = {}, exp_net_arg = {}, verbose=True, center_msg='out'):

        """\
        Initialization of a HERGAST trainer.

        Parameters
        ----------
        adata
            AnnData object of scanpy package.
        batch_data
            Using the DIC strategy or not? Used for large scale ST data (Visium HD, Xenium, etc)
        num_batch_x_y
            A tuple specifying the number of points at which to segment the spatially transcribed image on the x and y axes.
            Each split is then trained as a batch. Only useful when batch_data=True.
        spatial_net_arg
            A dict passing key-word arguments to calculating spatial network in each batch data. See `Cal_Spatial_Net`.
        exp_net_arg
            A dict passing key-word arguments to calculating expression network in each batch data. See `Cal_Expression_Net`
        device_idx
            An integer specifying which GPU to use (if multiple GPUs are available)
        center_msg
            In the message passing GNN mechanism:
            'out' : the message flows out from the center spot to neighboring spots.
            'in' : the messages flow into the center spot from neighboring spots.
        """

        if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")
        elif dim_reduction == 'HVG':
            if 'highly_variable' not in adata.var.keys():
                raise ValueError("HVG has not been computed! Run sc.pp.highly_variable_genes first!")
        else:
            warnings.warn("No dimentional reduction method specified, using all genes' expression as input.")

        self.dim_reduction = dim_reduction
        self.batch_data = batch_data
        self.adata = adata

        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        if 'Exp_Net' not in adata.uns.keys():
            raise ValueError("Exp_Net is not existed! Run Cal_Expression_Net first!")
        self.data = Transfer_pytorch_Data(adata, dim_reduction=dim_reduction,center_msg=center_msg)
        if verbose:
            print('Size of Input: ', self.data.x.shape)

        if batch_data:
            self.num_batch_x, self.num_batch_y = num_batch_x_y
            Batch_list = Batch_Data(adata, num_batch_x=self.num_batch_x, num_batch_y=self.num_batch_y)
            for temp_adata in Batch_list:
                Cal_Spatial_Net(temp_adata, **spatial_net_arg)
                Cal_Expression_Net(temp_adata, dim_reduce=dim_reduction, **exp_net_arg)
            data_list = [Transfer_pytorch_Data(adata, dim_reduction=dim_reduction,center_msg=center_msg) for adata in Batch_list]
            self.loader = DataLoader(data_list, batch_size=1, shuffle=True)

        self.device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
        self.model = None


    def train_HERGAST(self, save_path = None, hidden_dims=[100, 32], n_epochs=200, lr=0.001, 
                    key_added='HERGAST', att_drop = 0.3, gradient_clipping=5., weight_decay=0.0001,
                    random_seed=0, save_loss=False, save_reconstrction=True, save_attention=False):

        """\
        Training graph attention auto-encoder.

        Parameters
        ----------
        save_path
            directory to save the trained HERGAST model. Default not save the model.
        n_clusters
            number of clusters to set when calculating early stopping criterion.
        hidden_dims
            The dimension of the encoder (depends on HERGAST or RGAST2).
        n_epochs
            Number of total epochs in training.
        lr
            Learning rate for AdamOptimizer.
        key_added
            The latent embeddings are saved in adata.obsm[key_added].
        gradient_clipping
            Gradient Clipping.
        weight_decay
            Weight decay for AdamOptimizer.
        save_loss
            If True, the training loss is saved in adata.uns['RGAST_loss'].
        save_reconstrction
            If True, the reconstructed PCA profiles are saved in adata.layers['RGAST_ReX'].

        Returns
        -------
        AnnData
        """
        self.save_path = save_path

        # seed_everything()
        seed=random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


        if self.model is None:
            model = HERGAST(hidden_dims = [self.data.x.shape[1]] + hidden_dims, dim_reduce=self.dim_reduction, att_drop=att_drop).to(self.device)
        else:
            model = self.model.to(self.device)
            
        data = self.data.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_list = []

        with tqdm(range(n_epochs)) as tq:
            for _ in tq:
                if self.batch_data:
                    for batch in self.loader:
                        batch = batch.to(self.device)
                        model.train()
                        optimizer.zero_grad()
                        z, out, _, _ = model(batch.x, batch.edge_index, batch.edge_type)
                        loss = F.mse_loss(batch.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                        loss.backward()
                        loss_list.append(loss.item())
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        optimizer.step()

                else:
                    model.train()
                    optimizer.zero_grad()
                    z, out, _, _ = model(data.x, data.edge_index, data.edge_type)
                    loss = F.mse_loss(data.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    loss_list.append(loss.item())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()

        with torch.no_grad():
            if self.batch_data:
                model.to('cpu')
                model.eval()
                z, out, att1, att2 = model(data.x.cpu(), data.edge_index.cpu(), data.edge_type.cpu())
                model.to(self.device)
            else:
                model.eval()
                z, out, att1, att2 = model(data.x, data.edge_index, data.edge_type)

        HERGAST_rep = z.to('cpu').detach().numpy()
        self.adata.obsm[key_added] = HERGAST_rep
        if save_path is not None:
            torch.save(model,f'{save_path}/model.pth')

        if save_loss:
            self.adata.uns['HERGAST_loss'] = loss_list
        if save_reconstrction:
            ReX = out.to('cpu').numpy()
            if self.dim_reduction == 'HVG':
                idx = np.where(self.adata.X==0)
                ReX[idx] = 0
            self.adata.obsm['HERGAST_ReX'] = ReX
            
        if save_attention:
            self.adata.uns['att1'] = (att1[0].to('cpu').numpy(),att1[1].to('cpu').numpy())
            self.adata.uns['att2'] = (att2[0].to('cpu').numpy(),att2[1].to('cpu').numpy())

        self.model = model

    def load_model(self, path):
        self.model = torch.load(path, map_location=self.device)

    def save_model(self, path):
        torch.save(self.model,f'{path}/model.pth')

    @torch.no_grad()
    def process(self, gdata = None):
        if gdata is None:
            gdata = self.data
        self.model.to(self.device)
        self.model.eval()
        gdata = gdata.to(self.device)
        return self.model(gdata.x, gdata.edge_index, gdata.edge_type)

